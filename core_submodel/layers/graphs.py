import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.generic_utils import to_cuda
from .common import GRUStep, GatedFusion
from ..layers.attention import MultiHeadedAttention


class GraphNN(nn.Module):
    def __init__(self, config):
        super(GraphNN, self).__init__()
        print('[ Using {}-hop GraphNN ]'.format(config['graph_hops']))
        self.device = config['device']
        hidden_size = config['graph_hidden_size']
        self.hidden_size = hidden_size
        self.graph_type = config['graph_type']
        self.graph_hops = config['graph_hops']
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False)
        self.head_num = config['head_num']
        if self.graph_type == 'static':
            self.static_graph_mp = GraphMessagePassing(config)
            self.static_gru_step = GRUStep(hidden_size, hidden_size)
            self.graph_update = self.static_graph_update
        elif self.graph_type == 'dynamic':
            self.global_att = MultiHeadedAttention(self.head_num, self.hidden_size, config)
            self.graph_update = self.dynamic_graph_update
        elif self.graph_type == 'hybrid':
            self.static_graph_mp = GraphMessagePassing(config)
            self.static_gru_step = GRUStep(hidden_size, hidden_size)
            self.gated_fusion = GatedFusion(hidden_size)
            self.global_att = MultiHeadedAttention(self.head_num, self.hidden_size, config)
            self.graph_update = self.hybrid_graph_update
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(self.graph_type))

        print('[ Using graph type: {} ]'.format(self.graph_type))

    def forward(self, node_feature, edge_vec, adj, node_mask=None):
        graph_embedding = self.graph_update(node_feature, edge_vec, adj, node_mask=node_mask)
        return graph_embedding

    def static_graph_update(self, node_feature, edge_vec, adj, node_mask=None):
        node2edge, edge2node = adj
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)
        for _ in range(self.graph_hops):
            fw_agg_state = self.static_graph_mp.mp_func(node_feature, edge_vec, edge2node.transpose(1, 2),
                                                        node2edge.transpose(1, 2))
            node_feature = self.static_gru_step(node_feature, fw_agg_state)
        graph_embedding = self.graph_maxpool(node_feature, node_mask)
        return graph_embedding

    def dynamic_graph_update(self, node_feature, edge_vec, adj, node_mask=None):
        att_node_mask = node_mask.unsqueeze(1)
        node_feature = torch.relu(node_feature)
        node_feature = self.global_att(node_feature, node_feature, node_feature, att_node_mask)
        graph_embedding = self.graph_maxpool(node_feature, node_mask)
        return graph_embedding

    def hybrid_graph_update(self, node_feature, edge_vec, adj, node_mask=None):
        static_node_feature = node_feature
        dynamic_node_feature = node_feature
        graph_embedding_static = self.static_graph_update(static_node_feature, edge_vec, adj, node_mask=node_mask)
        graph_embedding_dynamic = self.dynamic_graph_update(dynamic_node_feature, edge_vec, adj, node_mask=node_mask)
        graph_embedding = self.gated_fusion(graph_embedding_static, graph_embedding_dynamic)
        return graph_embedding

    def graph_maxpool(self, node_state, node_mask=None):
        node_mask = node_mask.unsqueeze(-1)
        node_state = node_state * node_mask.float()
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(node_embedding_p, kernel_size=node_embedding_p.size(-1)).squeeze(-1)
        return graph_embedding


class GraphMessagePassing(nn.Module):
    def __init__(self, config):
        super(GraphMessagePassing, self).__init__()
        self.config = config
        hidden_size = config['graph_hidden_size']
        if config['message_function'] == 'edge_pair':
            self.linear_edge = nn.Linear(config['edge_embed_dim'], hidden_size, bias=False)
            self.mp_func = self.msg_pass
        elif config['message_function'] == 'no_edge':
            self.mp_func = self.msg_pass
        else:
            raise RuntimeError('Unknown message_function: {}'.format(config['message_function']))

    def msg_pass(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)                      # batch_size x num_edges x hidden_size
        if edge_vec is not None and self.config['message_function'] == 'edge_pair':
            node2edge_emb = node2edge_emb + self.linear_edge(edge_vec)
        agg_state = torch.bmm(edge2node, node2edge_emb)
        return agg_state