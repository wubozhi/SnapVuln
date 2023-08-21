import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .models.Graph2Vul import Graph2Vul
from .models.Seq2Vul import Seq2Vul
from .models.SelfAtt2Vul import SelfAtt2Vul
from .utils.vocab_utils import VocabModel
from .utils import constants as Constants
from sklearn.metrics import accuracy_score, classification_report, average_precision_score


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        self.config = config
        if config['model_name'] in ['Graph2Vul']:
            self.net_module = Graph2Vul
        elif config['model_name'] in ['Seq2Vul']:
            self.net_module = Seq2Vul
        elif config['model_name'] in ['SelfAtt2Vul']:
            self.net_module = SelfAtt2Vul
        else:
            raise RuntimeError('Unknown model_name: {}'.format(config['model_name']))
        print('[ Running {} model ]'.format(config['model_name']))

        self.vocab_model = VocabModel.build(self.config['saved_vocab_file'], train_set, config)
        self.config['num_edge_types'] = len(self.vocab_model.edge_vocab)
        if self.config['pretrained']:
            state_dict_opt = self.init_saved_network(self.config['pretrained'])
        else:
            assert train_set is not None
            # Building network.
            self._init_new_network()

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        print('#Parameters = {}\n'.format(num_params))

        self.criterion = nn.BCELoss()
        self._init_optimizer()

    def init_saved_network(self, saved_dir):
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        self.saved_epoch = saved_params.get('epoch', 0)

        word_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                              pretrained_vecs=self.vocab_model.word_vocab.embeddings)
        self.network = self.net_module(self.config, word_embedding, self.vocab_model.word_vocab)

        # Merge the arguments
        if state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

        return state_dict.get('optimizer', None) if state_dict else None

    def _init_new_network(self):
        word_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                              pretrained_vecs=self.vocab_model.word_vocab.embeddings)
        self.network = self.net_module(self.config, word_embedding, self.vocab_model.word_vocab)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adagrad':
            self.optimizer = optim.Adagrad(parameters, lr=self.config['learning_rate'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        return nn.Embedding(vocab_size, embed_size, padding_idx=0, _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    def save(self, dirname, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            'config': self.config,
            'dir': dirname,
            'epoch': epoch
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

    def predict(self, batch, step, update=True, mode='train'):
        self.network.train(update)

        if mode == 'train':
            loss, loss_value, batch_labels, batch_probs = train_batch(batch, self.network, self.criterion)
            # Accumulate gradients
            loss = loss / self.config['grad_accumulated_steps']  # Normalize our loss (if averaged)
            # Run backward
            loss.backward()

            if (step + 1) % self.config['grad_accumulated_steps'] == 0:  # Wait for several backward steps
                if self.config['grad_clipping']:
                    # Clip gradients
                    parameters = [p for p in self.network.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])
                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
            output = {
                'loss': loss_value,
                'probs': batch_probs
            }
        elif mode == 'dev':
            loss_value, batch_labels, batch_probs = dev_batch(batch, self.network, criterion=self.criterion)
            output = {
                'loss': loss_value,
                'probs': batch_probs
            }
        else:
            batch_labels, batch_probs = test_batch(batch, self.network, criterion=self.criterion)
            output = {
                'probs': batch_probs
            }
        return output


def train_batch(batch, network, criterion):
    network.train(True)
    with torch.set_grad_enabled(True):
        network_out = network(batch, criterion)
        loss = network_out.loss
        loss_value = network_out.loss_value
    return loss, loss_value, network_out.labels, network_out.probs


# Development phase
def dev_batch(batch, network, criterion=None):
    """Test the `network` on the `batch`, return the ROUGE score and the loss."""
    network.train(False)
    network_out = eval_decode_batch(batch, network, criterion)
    loss_value = network_out.loss_value
    return loss_value, network_out.labels, network_out.probs


# Test phase
def test_batch(batch, network, criterion=None):
    """Test the `network` on the `batch`, return the ROUGE score and the loss."""
    network.train(False)
    network_out = eval_decode_batch(batch, network, criterion)
    return network_out.labels, network_out.probs


def eval_decode_batch(batch, network, criterion=None):
    """Test the `network` on the `batch`, return the decoded textual tokens and the Output."""
    with torch.no_grad():
        out = network(batch, criterion)
    return out


def evaluate_predictions(labels, probs):
    predicted_labels = (probs > 0.5).float()
    labels = labels.cpu().data.numpy()
    predicted_labels = predicted_labels.cpu().data.numpy()
    acc = accuracy_score(labels, predicted_labels)
    reports = classification_report(labels, predicted_labels, target_names=['0', '1'], output_dict=True)
    neg_pre = reports['0']['precision']
    neg_rec = reports['0']['recall']
    neg_f1 = reports['0']['f1-score']
    pos_pre = reports['1']['precision']
    pos_rec = reports['1']['recall']
    pos_f1 = reports['1']['f1-score']
    prc_auc = average_precision_score(labels, probs.cpu().data.numpy())
    metrics = {'Acc': acc, 'Neg_pre': neg_pre, 'Neg_rec': neg_rec, 'Neg_f1': neg_f1, 'Pos_pre': pos_pre,
               'Pos_rec': pos_rec, 'Pos_f1': pos_f1, 'PRC_AUC': prc_auc}
    return metrics