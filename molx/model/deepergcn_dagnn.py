import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, GENConv, DeepGCNLayer
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def make_mask(batch, device):
    n = batch.shape[0]
    mask = torch.eq(batch.unsqueeze(1), batch.unsqueeze(0))
    mask = (torch.ones((n, n)) - torch.eye(n)).to(device) * mask
    count = torch.sum(mask)
    return mask, count

class Deepergcn_dagnn_dist(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, 
                 JK="last", aggr='softmax', norm='batch', 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Deepergcn_dagnn_dist, self).__init__()

        self.deepergcn_dagnn = DeeperDAGNN_node_Virtualnode(num_layers, emb_dim, drop_ratio, JK, aggr, norm)
        self.calc_dist = DistMax(emb_dim)

    def forward(self, batched_data, train=False):
        xs = self.deepergcn_dagnn(batched_data)
        mask_d_pred, mask, count = self.calc_dist(xs, batched_data.batch)
        return mask_d_pred, mask, count


class DistMax(torch.nn.Module):
    def __init__(self, emb_dim, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(DistMax, self).__init__()
        self.fc = torch.nn.Linear(in_features=emb_dim, out_features=1)
        self.device = device

    def forward(self, xs, batch, train=False):
        d_pred = self.fc(torch.max(xs.unsqueeze(0), xs.unsqueeze(1))).squeeze()
        mask, count = make_mask(batch, self.device)

        if train:
            mask_d_pred = d_pred * mask
        else:
            mask_d_pred = F.relu(d_pred * mask)
        return mask_d_pred, mask, count


class Deepergcn_dagnn_coords(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5,
                 JK="last", aggr='softmax', norm='batch',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Deepergcn_dagnn_coords, self).__init__()

        self.deepergcn_dagnn = DeeperDAGNN_node_Virtualnode(num_layers, emb_dim, drop_ratio, JK, aggr, norm)
        self.calc_dist = DistCoords(emb_dim)

    def forward(self, batched_data, train=False):
        xs = self.deepergcn_dagnn(batched_data)
        mask_d_pred, mask, count = self.calc_dist(xs, batched_data.batch)
        return mask_d_pred, mask, count


class DistCoords(torch.nn.Module):
    def __init__(self, emb_dim, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(DistCoords, self).__init__()
        self.fc = torch.nn.Linear(in_features=emb_dim, out_features=3)
        self.device = device

    def forward(self, xs, batch, train=False):
        xs = self.fc(xs)
        d_pred = torch.cdist(xs, xs)
        mask, count = make_mask(batch, self.device)

        if train:
            mask_d_pred = d_pred * mask
        else:
            mask_d_pred = F.relu(d_pred * mask)
        return mask_d_pred, mask, count


class DAGNN(MessagePassing):
    def __init__(self, K, emb_dim, normalize=True, add_self_loops=True):
        super(DAGNN, self).__init__()
        self.K = K
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.proj = torch.nn.Linear(emb_dim, 1)

        self._cached_edge_index = None

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize:
            edge_index, norm = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)

        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)

        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

    def reset_parameters(self):
        self.proj.reset_parameters()


class DeeperDAGNN_node_Virtualnode(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", aggr='softmax', norm='batch'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(DeeperDAGNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # self.input_encode_manner = input_encode_manner

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        self.dagnn = DAGNN(5, emb_dim)

        ###List of GNNs
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=True, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)


            layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.layers.append(layer)

        for layer in range(num_layers - 1):
            if norm=="batch":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                                     torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))
            elif norm=="layer":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU(), \
                                                                     torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU()))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        edge_attr = self.bond_encoder(edge_attr)
        h = self.atom_encoder(x)

        h_list = []

        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h = h + virtualnode_embedding[batch]
        h = self.layers[0].conv(h, edge_index, edge_attr)

        h_list.append(h)
        for i, layer in enumerate(self.layers[1:]):
            h = layer(h, edge_index, edge_attr)

            ### update the virtual nodes
            ### add message from graph nodes to virtual nodes
            virtualnode_embedding_temp = global_add_pool(h, batch) + virtualnode_embedding

            ### transform virtual nodes using MLP
            virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

            h = h + virtualnode_embedding[batch]
            h_list.append(h)

        h = self.layers[0].act(self.layers[0].norm(h))
        h = F.dropout(h, p=0, training=self.training)

        h_list.append(h)
        h = h + virtualnode_embedding[batch]

        h = self.dagnn(h, edge_index)
        h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]

        return node_representation
