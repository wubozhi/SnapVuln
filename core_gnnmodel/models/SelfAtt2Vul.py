from ..layers.common import dropout
from ..layers.attention import *
from ..utils.generic_utils import create_mask


class Output(object):

    def __init__(self, labels=0, loss=0, loss_value=0, probs=0):
        self.labels = labels
        self.loss = loss  # scalar
        self.loss_value = loss_value  # float value, excluding coverage loss
        self.probs = probs


class SelfAtt2Vul(nn.Module):

    def __init__(self, config, word_embedding, word_vocab):
        """
        :param word_vocab: mainly for info about special tokens and word_vocab size
        :param config: model hyper-parameters
        :param max_dec_steps: max num of decoding steps (only effective at test time, as during
                              training the num of steps is determined by the `target_tensor`); it is
                              safe to change `self.max_dec_steps` as the network architecture is
                              independent of src/tgt seq lengths
        Create the graph2seq model; its encoder and decoder will be created automatically.
        """
        super(SelfAtt2Vul, self).__init__()
        self.name = 'SelfAtt2Vul'
        self.device = config['device']
        self.word_dropout = config['word_dropout']
        self.word_vocab = word_vocab
        self.vocab_size = len(word_vocab)
        self.rnn_type = config['rnn_type']
        self.model_name = config['model_name']
        self.enc_hidden_size = config['enc_hidden_size']
        self.word_embed = word_embedding
        self.message_function = config['message_function']
        self.node_initial_type = config['node_initialize_type']
        if config['fix_word_embed']:
            print('[ Fix word embeddings ]')
            for param in self.word_embed.parameters():
                param.requires_grad = False
        self.edge_embed = nn.Embedding(config['num_edge_types'], config['edge_embed_dim'], padding_idx=0)
        self.self_att = MultiHeadedAttention(config['head_num'], self.enc_hidden_size, config)
        self.out_logits = nn.Linear(self.enc_hidden_size, 1, bias=False)
        self.out_act = nn.Sigmoid()

    def forward(self, ex, criterion=None):
        encoder_token_embedded = self.word_embed(ex['srcs'])
        labels = ex['targets']
        encoder_token_embedded = dropout(encoder_token_embedded, self.word_dropout, shared_axes=[-2],
                                         training=self.training)
        max_code_lens = torch.max(ex['src_lens']).cpu().detach().numpy()
        code_sequence_embedded_mask = create_mask(ex['src_lens'], max_code_lens, self.device)
        token_feature_weight = self.self_att(encoder_token_embedded, encoder_token_embedded, encoder_token_embedded,
                                             code_sequence_embedded_mask.unsqueeze(1))
        seq_repr = torch.sum(token_feature_weight, dim=1)
        r = Output()
        logits = self.out_logits(seq_repr)
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