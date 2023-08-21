from ..layers.common import EncoderRNN, dropout
from ..layers.attention import *


class Output(object):

    def __init__(self, labels=0, loss=0, loss_value=0, probs=0):
        self.labels = labels
        self.loss = loss  # scalar
        self.loss_value = loss_value  # float value, excluding coverage loss
        self.probs = probs


class Seq2Vul(nn.Module):

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
        super(Seq2Vul, self).__init__()
        self.name = 'Seq2Vul'
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
        self.sequence_encoder = EncoderRNN(config['word_embed_dim'], self.enc_hidden_size,
                                           bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'],
                                           rnn_type=self.rnn_type, rnn_dropout=config['enc_rnn_dropout'],
                                           device=self.device)
        self.out_logits = nn.Linear(self.enc_hidden_size, 1, bias=False)
        self.out_act = nn.Sigmoid()

    def forward(self, ex, criterion=None):
        encoder_token_embedded = self.word_embed(ex['srcs'])
        labels = ex['targets']
        encoder_token_embedded = dropout(encoder_token_embedded, self.word_dropout, shared_axes=[-2],
                                         training=self.training)
        sequence_initial_output, sequence_initial_representation = self.sequence_encoder(encoder_token_embedded,
                                                                                         ex['src_lens'])
        r = Output()
        logits = self.out_logits(sequence_initial_representation[0].squeeze(0))
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