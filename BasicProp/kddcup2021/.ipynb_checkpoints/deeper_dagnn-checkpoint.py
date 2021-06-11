import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, GENConv, DeepGCNLayer
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_scatter import scatter_mean



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
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(DeeperDAGNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.graph_pooling = graph_pooling



        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        
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
            
            
            
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
        
        

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        edge_attr = self.bond_encoder(edge_attr)

        h_list = []

        h = self.atom_encoder(x)
        
         
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
                
        h_graph = self.pool(node_representation, batched_data.batch)
        output = self.graph_pred_linear(h_graph)
       

        if self.training:
            return output
        else:
            ### At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)