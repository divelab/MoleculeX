import torch
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_softmax, scatter_add, scatter_sum
from torch_geometric.nn import MessagePassing, GCNConv, NNConv, GINConv
from torch_geometric.utils import degree

import numpy as np
    
    
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, activation):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.activation = activation
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x
    

    
class SizeNorm(torch.nn.Module):
    def __init__(self):
        super(SizeNorm, self).__init__()

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        inv_sqrt_deg = degree(batch, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg[batch].view(-1, 1)
   
    

class BatchNorm(torch.nn.BatchNorm1d):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm, self).__init__(in_channels, eps, momentum, affine,
                                        track_running_stats)

    def forward(self, x):
        return super(BatchNorm, self).forward(x)


    def __repr__(self):
        return ('{}({}, eps={}, momentum={}, affine={}, '
                'track_running_stats={})').format(self.__class__.__name__,
                                                  self.num_features, self.eps,
                                                  self.momentum, self.affine,
                                                  self.track_running_stats)

    
    
    
###########################################
class EdgeModel_ml2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(EdgeModel_ml2, self).__init__()
        self.edge_mlp = MLP(input_dim, hidden_dim, output_dim, dropout, F.relu)
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr, u[batch[row]]], 1)
        return self.edge_mlp(out)      ### Step 1

class NodeModel_ml2(torch.nn.Module):
    def __init__(self, input_dim1, hidden_dim1, output_dim1, input_dim2, hidden_dim2, output_dim2, dropout):
        super(NodeModel_ml2, self).__init__()
        self.node_mlp_1 = MLP(input_dim1, hidden_dim1, output_dim1, dropout, F.relu)
        self.node_mlp_2 = MLP(input_dim2, hidden_dim2, output_dim2, dropout, F.relu)
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = scatter_sum(out, col, dim=0, dim_size=x.size(0))   
        out = self.node_mlp_1(out)  ### Step 2
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)  ### Step 3
  
class SubgraphModel_ml2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(SubgraphModel_ml2, self).__init__()
        self.mlp1 = MLP(hidden_dim, hidden_dim, hidden_dim, dropout, F.relu)
        self.mlp2 = MLP(hidden_dim, hidden_dim, hidden_dim, dropout, F.relu)
        self.subgraph_mlp = MLP(input_dim, hidden_dim, output_dim, dropout, F.relu)
       

    def forward(self, x, x_clique, tree_edge_index, atom2clique_index, u, tree_batch):
        row, col = tree_edge_index
        out = scatter_sum(x_clique[row], col, dim=0, dim_size=x_clique.size(0)) 
        out = self.mlp1(out)
        row_assign, col_assign = atom2clique_index
        node_info = scatter_sum(x[row_assign], col_assign, dim=0, dim_size=x_clique.size(0))
        node_info = self.mlp2(node_info)  ### Step 4
        out = torch.cat([node_info, x_clique, out, u[tree_batch]], dim=1)
        return self.subgraph_mlp(out)  ### Step 5
   


class GlobalModel_ml2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GlobalModel_ml2, self).__init__()
        self.global_mlp = MLP(input_dim, hidden_dim, output_dim, dropout, F.relu)


    def forward(self, x, edge_index, edge_attr, x_clique, u, batch, tree_batch):
        row, col = edge_index
        edge_info = scatter_mean(edge_attr, batch[row], dim=0, dim_size=u.size(0)) ### Step 6
        node_info = scatter_mean(x, batch, dim=0, dim_size=u.size(0))   ### Step 7
        subgraph_info = scatter_mean(x_clique, tree_batch, dim=0, dim_size=u.size(0))  ### Step 8
        out = torch.cat([u, node_info, edge_info, subgraph_info], dim=1) 
        return self.global_mlp(out)  ### Step 9

    
    
class MetaLayer_ml2(torch.nn.Module):
    def __init__(self, input_node_rep_dim, input_edge_rep_dim, input_subgraph_rep_dim, input_global_rep_dim, output_node_rep_dim, output_edge_rep_dim, output_subgraph_rep_dim, output_global_rep_dim, hidden, dropout):
        super(MetaLayer_ml2, self).__init__()
        self.edge_model = EdgeModel_ml2(2*input_node_rep_dim+input_edge_rep_dim+input_global_rep_dim, hidden, output_edge_rep_dim, dropout)
        self.node_model = NodeModel_ml2(input_node_rep_dim+output_edge_rep_dim, hidden, hidden, hidden+input_node_rep_dim+input_global_rep_dim, hidden, output_node_rep_dim, dropout)
        self.subgraph_model = SubgraphModel_ml2(2*input_subgraph_rep_dim+output_node_rep_dim+input_global_rep_dim, hidden, output_subgraph_rep_dim, dropout)
        self.global_model = GlobalModel_ml2(input_global_rep_dim+output_node_rep_dim+output_edge_rep_dim+output_subgraph_rep_dim, hidden, output_global_rep_dim, dropout)
       

    def forward(self, x, edge_index, edge_attr, u, tree_edge_index, atom2clique_index, x_clique, ori_batch, tree_batch):

        edge_attr = self.edge_model(x, edge_index, edge_attr, u, ori_batch) 
        x = self.node_model(x, edge_index, edge_attr, u, ori_batch)        
        x_clique = self.subgraph_model(x, x_clique, tree_edge_index, atom2clique_index, u, tree_batch)                    
        u = self.global_model(x, edge_index, edge_attr, x_clique, u, ori_batch, tree_batch)
        
        return x, edge_attr, x_clique, u

    
    
class MLNet2(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features, hidden, dropout, num_tasks, depth, graph_level_feature):
        super(MLNet2, self).__init__()
        self.mlp_node = MLP(num_node_features, hidden, hidden, dropout, F.relu)
        self.mlp_edge = MLP(num_edge_features, hidden, hidden, dropout, F.relu)
        self.emb_subgraph = torch.nn.Embedding(4, hidden)
        self.graph_level_feature = graph_level_feature
        if self.graph_level_feature:
            self.mlp_global = MLP(num_edge_features, hidden, hidden, dropout, F.relu)
            self.mlp1 = MLP(num_global_features+hidden, hidden, num_tasks, dropout, F.relu)
        else:
            self.mlp_global = MLP(num_edge_features, hidden, hidden, dropout, F.relu)
            self.mlp1 = MLP(hidden, hidden, num_tasks, dropout, F.relu)
        

        self.dropout = dropout
        self.num_global_features = num_global_features
        
        self.depth = depth
        
        self.gn = torch.nn.ModuleList([MetaLayer_ml2(hidden, hidden, hidden, hidden, hidden, hidden, hidden, hidden, hidden, dropout) for i in range(self.depth)])
        self.norm_node = torch.nn.ModuleList([SizeNorm() for i in range(self.depth+1)])
        self.norm_edge = torch.nn.ModuleList([SizeNorm() for i in range(self.depth+1)])
        self.norm_subgraph = torch.nn.ModuleList([SizeNorm() for i in range(self.depth+1)])
        self.bn_node = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth+1)])
        self.bn_edge = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth+1)])
        self.bn_subgraph = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth+1)])
        self.bn_global = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth+1)])
        


    def forward(self, batch_data):
        
        if self.graph_level_feature:  ### Use rdkit_2d_normalized_features
            x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
            row, col = edge_index
            u = scatter_mean(edge_attr, batch_data.batch[row], dim=0, dim_size=max(batch_data.batch)+1)
            tree_edge_index, atom2clique_index, num_cliques, x_clique = batch_data.tree_edge_index, batch_data.atom2clique_index, batch_data.num_cliques, batch_data.x_clique
            aug_feat = batch_data.graph_attr
            if len(aug_feat.shape) != 2:
                aug_feat = torch.reshape(aug_feat, (-1, self.num_global_features))
        else:
            x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
            row, col = edge_index
            u = scatter_mean(edge_attr, batch_data.batch[row], dim=0, dim_size=max(batch_data.batch)+1)
            tree_edge_index, atom2clique_index, num_cliques, x_clique = batch_data.tree_edge_index, batch_data.atom2clique_index, batch_data.num_cliques, batch_data.x_clique
            
        x = self.mlp_node(x)
        edge_attr = self.mlp_edge(edge_attr)
        x_clique = self.emb_subgraph(x_clique)
        u = self.mlp_global(u)

        row, col = edge_index
        
        ori_batch = batch_data.batch
        tree_batch = torch.repeat_interleave(num_cliques)
        
        x = self.norm_node[-1](x, ori_batch)
        edge_attr = self.norm_edge[-1](edge_attr, ori_batch[row])
        x_clique = self.norm_subgraph[-1](x_clique, tree_batch)
        x = self.bn_node[-1](x)
        edge_attr = self.bn_edge[-1](edge_attr)
        x_clique = self.bn_subgraph[-1](x_clique)
        u = self.bn_global[-1](u)
        
        for i in range(self.depth):

            x, edge_attr, x_clique, u = self.gn[i](x, edge_index, edge_attr, u, tree_edge_index, atom2clique_index, x_clique, ori_batch, tree_batch)
            
            x = self.norm_node[i](x, batch_data.batch)
            edge_attr = self.norm_edge[i](edge_attr, batch_data.batch[row])
            x_clique = self.norm_subgraph[i](x_clique, tree_batch)
            x = self.bn_node[i](x)
            edge_attr = self.bn_edge[i](edge_attr)
            x_clique = self.bn_subgraph[i](x_clique)
            u = self.bn_global[i](u)

        if self.graph_level_feature:
            u = torch.cat([u,aug_feat], dim=1)
        out = self.mlp1(u)
        

        return out 
    

    
    
############## Ablation studey: w/o subgraph-level   

class GlobalModel_ml3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GlobalModel_ml3, self).__init__()
        self.global_mlp = MLP(input_dim, hidden_dim, output_dim, dropout, F.relu)


    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        edge_info = scatter_mean(edge_attr, batch[row], dim=0, dim_size=u.size(0)) 
        node_info = scatter_mean(x, batch, dim=0, dim_size=u.size(0))   
        out = torch.cat([u, node_info, edge_info], dim=1) 
        return self.global_mlp(out)
    
class MetaLayer_ml3(torch.nn.Module):
    def __init__(self, input_node_rep_dim, input_edge_rep_dim, input_global_rep_dim, output_node_rep_dim, output_edge_rep_dim, output_global_rep_dim, hidden, dropout):
        super(MetaLayer_ml3, self).__init__()
        self.edge_model = EdgeModel_ml2(2*input_node_rep_dim+input_edge_rep_dim+input_global_rep_dim, hidden, output_edge_rep_dim, dropout)
        self.node_model = NodeModel_ml2(input_node_rep_dim+output_edge_rep_dim, hidden, hidden, hidden+input_node_rep_dim+input_global_rep_dim, hidden, output_node_rep_dim, dropout)
        self.global_model = GlobalModel_ml3(input_global_rep_dim+output_node_rep_dim+output_edge_rep_dim, hidden, output_global_rep_dim, dropout)
       

    def forward(self, x, edge_index, edge_attr, u, ori_batch):

        edge_attr = self.edge_model(x, edge_index, edge_attr, u, ori_batch) 
        x = self.node_model(x, edge_index, edge_attr, u, ori_batch)            
        u = self.global_model(x, edge_index, edge_attr, u, ori_batch)
        
        return x, edge_attr, u
    

    
class MLNet3(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features, hidden, dropout, num_tasks, depth, graph_level_feature):
        super(MLNet3, self).__init__()
        self.mlp_node = MLP(num_node_features, hidden, hidden, dropout, F.relu)
        self.mlp_edge = MLP(num_edge_features, hidden, hidden, dropout, F.relu)
        self.graph_level_feature = graph_level_feature
        if self.graph_level_feature:
            self.mlp_global = MLP(num_edge_features, hidden, hidden, dropout, F.relu)
            self.mlp1 = MLP(num_global_features+hidden, hidden, num_tasks, dropout, F.relu)
        else:
            self.mlp_global = MLP(num_edge_features, hidden, hidden, dropout, F.relu)
            self.mlp1 = MLP(hidden, hidden, num_tasks, dropout, F.relu)
        

        self.dropout = dropout
        self.num_global_features = num_global_features
        
        self.depth = depth
        
        self.gn = torch.nn.ModuleList([MetaLayer_ml3(hidden, hidden, hidden, hidden, hidden, hidden, hidden, dropout) for i in range(self.depth)])
        self.norm_node = torch.nn.ModuleList([SizeNorm() for i in range(self.depth+1)])
        self.norm_edge = torch.nn.ModuleList([SizeNorm() for i in range(self.depth+1)])
        self.norm_subgraph = torch.nn.ModuleList([SizeNorm() for i in range(self.depth+1)])
        self.bn_node = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth+1)])
        self.bn_edge = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth+1)])
        self.bn_subgraph = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth+1)])
        self.bn_global = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth+1)])
        


    def forward(self, batch_data):
        
        if self.graph_level_feature:  ### Use rdkit_2d_normalized_features as input graph-level feature
            x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
            row, col = edge_index
            u = scatter_mean(edge_attr, batch_data.batch[row], dim=0, dim_size=max(batch_data.batch)+1)
            aug_feat = batch_data.graph_attr
            if len(aug_feat.shape) != 2:
                aug_feat = torch.reshape(aug_feat, (-1, self.num_global_features))
        else:
            x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
            row, col = edge_index
            u = scatter_mean(edge_attr, batch_data.batch[row], dim=0, dim_size=max(batch_data.batch)+1)
            
        x = self.mlp_node(x)
        edge_attr = self.mlp_edge(edge_attr)
        u = self.mlp_global(u)

        row, col = edge_index
        
        ori_batch = batch_data.batch
        
        x = self.norm_node[-1](x, ori_batch)
        edge_attr = self.norm_edge[-1](edge_attr, ori_batch[row])
        x = self.bn_node[-1](x)
        edge_attr = self.bn_edge[-1](edge_attr)
        u = self.bn_global[-1](u)
               
        for i in range(self.depth):

            x, edge_attr, u = self.gn[i](x, edge_index, edge_attr, u, ori_batch)
            
            x = self.norm_node[i](x, batch_data.batch)
            edge_attr = self.norm_edge[i](edge_attr, batch_data.batch[row])
            x = self.bn_node[i](x)
            edge_attr = self.bn_edge[i](edge_attr)
            u = self.bn_global[i](u)

        if self.graph_level_feature:
            u = torch.cat([u,aug_feat], dim=1)
        out = self.mlp1(u)
        

        return out 
        
    
