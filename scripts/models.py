import numpy

import sklearn
import torch
import torch.nn as nn
import dgl
import dgllife
from dgllife.model import MPNNGNN

def get_diff(ratom_feats, patom_feats, atom_p2r):
    diff_feats = [patom_feats[pidx]-ratom_feats[ridx] for pidx, ridx in atom_p2r.items()]
    return torch.stack(diff_feats)

def batch_sum(bg, diff_feats):
    sum_feats = []
    node_cnt = 0
    graphs = dgl.unbatch(bg)
    for graph in graphs:
        sum_feats.append(torch.sum(diff_feats[node_cnt:node_cnt+graph.number_of_nodes()], dim = 0))
        node_cnt += graph.number_of_nodes()
    return torch.stack(sum_feats)
    
class LocalDiff(nn.Module):
    def __init__(self,
                 node_in_feats = 74,
                 edge_in_feats = 13,
                 node_out_feats = 128,
                 edge_hidden_feats = 32,
                 num_step_message_passing = 3):
        super(LocalDiff, self).__init__()
                
        self.activation = nn.ReLU()
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.linear_diff = nn.Sequential(
                                nn.Linear(node_out_feats, node_out_feats*2),
                                self.activation, 
                                nn.Dropout(0.2),
                                nn.Linear(node_out_feats*2, node_out_feats))
        
        self.linear_ea = nn.Sequential(
                                nn.Linear(node_out_feats, node_out_feats),
                                self.activation, 
                                nn.Dropout(0.2),
                                nn.Linear(node_out_feats, 1))

    def forward(self, g, node_feats, edge_feats, atom_p2r):
        rg, pg = g
        rnode_feats, pnode_feats = node_feats
        redge_feats, pedge_feats = edge_feats
        
        ratom_feats = self.mpnn(rg, rnode_feats, redge_feats)
        patom_feats = self.mpnn(pg, pnode_feats, pedge_feats)
        diff_feats = get_diff(ratom_feats, patom_feats, atom_p2r)
        diff_feats = self.linear_diff(diff_feats)
        ea_feats = batch_sum(pg, diff_feats)
        ea = self.linear_ea(ea_feats)
        return ea.view(-1)


