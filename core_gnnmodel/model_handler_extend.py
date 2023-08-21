import torch
import torch.backends.cudnn as cudnn
from .model import Model, evaluate_predictions
from .utils.data_utils import DataStream, read_all_Datasets
from .utils import Timer, DummyLogger, AverageMeter
from .model_handler import ModelHandler_GNN
import pandas as pd


class ModelHandlerExtend_GNN(ModelHandler_GNN):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """
    def __init__(self, config):
        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        self._dev_loss = AverageMeter()

        self.model = Model(config, None)
        self.model.network = self.model.network.to(self.device)
        self.config = self.model.config
        self.is_test = False

    def test(self):
        self.is_test = True
        test_set, test_src_len = read_all_Datasets(self.config['testset'], self.config, mode='dev', isLower=True)
        print('# of testing examples: {}'.format(len(test_set)))
        print('Test source node length: {}'.format(test_src_len))
        self.test_loader = DataStream(test_set, self.model.vocab_model.word_vocab, self.model.vocab_model.edge_vocab,
                                      config=self.config, isShuffle=False, isLoop=False, isSort=True,
                                      batch_size=self.config['test_batch_size'])
        self._n_test_batches = self.test_loader.get_num_batch()
        self._n_test_examples = len(test_set)
        timer = Timer("Test")
        for param in self.model.network.parameters():
            param.requires_grad = False
        labels, probs, files = self._run_epoch(self.test_loader, training=False, verbose=0)
        self.write_test_results(labels, probs, files, self.config['result'])
        metrics = evaluate_predictions(labels, probs)
        format_str = '<<<<<<<<<<<<<<<<< Test >>>>>>>>>>>>>>>>>'
        format_str += '-- Loss: {:0.5f}'.format(self._dev_loss.mean())
        format_str += self.plain_metric_to_str(metrics)
        self.logger.write_to_file(format_str)
        print(format_str)
        timer.finish()
        self.logger.close()
        return metrics

    def write_test_results(self, labels, probs, files, out_file):
        records = []
        predicted_labels = (probs > 0.5).float()
        labels = labels.cpu().data.numpy()
        probs = probs.cpu().data.numpy()
        predicted_labels = predicted_labels.cpu().data.numpy()
        for index in range(len(labels)):
            records.append({'label': labels[index], 'prob': probs[index], 'predicted_label': predicted_labels[index],
                            'file': files[index]})
        df = pd.DataFrame(records)
        df.to_csv(out_file +'_test_detailed_results.csv', index=False,
                  columns=['label', 'prob', 'predicted_label', 'file'])
