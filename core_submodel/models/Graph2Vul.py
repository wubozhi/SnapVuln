from ..layers.common import EncoderRNN, dropout
from ..layers.attention import *
from ..layers.graphs import GraphNN
from ..utils.generic_utils import create_mask
import torch.nn as nn


class Output(object):

    def __init__(self, labels=0, loss=0, loss_value=0, probs=0):
        self.labels = labels
        self.loss = loss  # scalar
        self.loss_value = loss_value  # float value, excluding coverage loss
        self.probs = probs


class Graph2Vul(nn.Module):

    def __init__(self, config, word_embedding, word_vocab):
        super(Graph2Vul, self).__init__()
        self.name = 'Graph2Vuln'
        self.device = config['device']
        self.max_slices_num = config['max_slices_num']
        self.word_dropout = config['word_dropout']
        self.rnn_type = config['rnn_type']
        self.model_name = config['model_name']
        self.enc_hidden_size = config['enc_hidden_size']
        self.word_embed = word_embedding
        self.message_function = config['message_function']
        self.node_initial_type = config['node_initialize_type']
        self.edge_embed = nn.Embedding(config['num_edge_types'], config['edge_embed_dim'], padding_idx=0)
        if config['fix_word_embed']:
            print('[ Fix word embeddings ]')
            for param in self.word_embed.parameters():
                param.requires_grad = False
        if self.node_initial_type == 'lstm':
            self.sequence_encoder = EncoderRNN(config['word_embed_dim'], self.enc_hidden_size,
                                               bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'],
                                               rnn_type=self.rnn_type, rnn_dropout=config['enc_rnn_dropout'],
                                               device=self.device)
        elif self.node_initial_type == 'self_att':
            self.self_att = MultiHeadedAttention(config['head_num'], self.enc_hidden_size, config)
        self.graph_encoder = GraphNN(config)
        self.multihead_attn = nn.MultiheadAttention(self.enc_hidden_size,1,batch_first=True)
        self.out_logits = nn.Linear(self.enc_hidden_size, 1, bias=False)
        self.out_act = nn.Sigmoid()

    def pad_node_initial_represenation(self, node_initial_representation, input_graphs):
        node_num = input_graphs['node_num']
        node_initial_representation = torch.split(node_initial_representation, node_num, dim=0)
        resize_node_initial_representation = torch.nn.utils.rnn.pad_sequence(node_initial_representation, batch_first=True)
        return resize_node_initial_representation

    def forward(self, ex, criterion=None):
        code_graphs = ex['code_graphs']
        slices_mask = ex['slices_mask']
        batch_size = ex['batch_size']
        labels = ex['targets']
        files = ex['files']
        if self.message_function == 'edge_mm':
            edge_vec = code_graphs['edge_features']
        else:
            edge_vec = self.edge_embed(code_graphs['edge_features'])
        node_word_index = code_graphs['node_word_index']
        node_word_lengths = code_graphs['node_word_lengths']
        encoder_node_token_embedded = self.word_embed(node_word_index)
        encoder_node_token_embedded = dropout(encoder_node_token_embedded, self.word_dropout, shared_axes=[-2],
                                              training=self.training)
        if self.node_initial_type == 'lstm':
            node_initial_output, node_initial_representation = self.sequence_encoder(encoder_node_token_embedded,
                                                                                     node_word_lengths)
            node_initial_representation_h = node_initial_representation[0].squeeze(0)
            resize_node_initial_representation_h = self.pad_node_initial_represenation(
                node_initial_representation_h, code_graphs)
        elif self.node_initial_type == 'sum':
            resize_node_initial_representation_h = torch.sum(encoder_node_token_embedded, dim=1)
            resize_node_initial_representation_h = self.pad_node_initial_represenation(
                resize_node_initial_representation_h, code_graphs)
        elif self.node_initial_type == 'mean':
            resize_node_initial_representation_h = torch.sum(encoder_node_token_embedded, dim=1) / \
                                                   node_word_lengths.unsqueeze(-1).type(torch.cuda.FloatTensor)
            resize_node_initial_representation_h = self.pad_node_initial_represenation(
                resize_node_initial_representation_h, code_graphs)
        else:
            raise RuntimeError('Unknown node initialization type: {}'.format(self.node_initial_type))
        node_length_masks = code_graphs['node_num_masks']
        node_features = resize_node_initial_representation_h
        # graph_embedding: [batch_size, max_slice_num, graph_emb_size]
        graph_embedding = self.graph_encoder(node_features, edge_vec,
                                             (code_graphs['node2edge'], code_graphs['edge2node']),
                                             node_mask=node_length_masks).view(batch_size, self.max_slices_num, -1)
        # print(batch_size, self.max_slices_num,self.enc_hidden_size)
        # print(graph_embedding.shape)
        # print(slices_mask)
        clss = torch.randn(graph_embedding.shape[0],1,self.enc_hidden_size).to(self.device)
        concat_clss_graph_embedding = torch.cat((clss, graph_embedding), axis=1)
        attn_emb, attn_map = self.multihead_attn(concat_clss_graph_embedding, concat_clss_graph_embedding, concat_clss_graph_embedding,key_padding_mask=slices_mask)
        # attn_emb, attn_map = self.multihead_attn(concat_clss_graph_embedding, concat_clss_graph_embedding, concat_clss_graph_embedding)
        # print(attn_map)
        graph_emb = attn_emb[:, 0, :]
        # print(graph_emb.shape)
        # graph_emb = torch.sum(graph_embedding, dim=1) 
        # with open("attention.txt","a") as f:
        #     f.write("file:\n")
        #     f.write(files)
        #     f.write("code_graphs:")
        r = Output()
        logits = self.out_logits(graph_emb)
        probs = self.out_act(logits).squeeze()
        nll_loss = self.BCE_loss(probs, labels, criterion)
        r.loss = torch.sum(nll_loss)
        r.loss_value = r.loss.item()
        r.probs = probs
        r.labels = labels
        return r

    def BCE_loss(self, prob, labels, criterion):
        loss = criterion(prob, labels)
        return loss