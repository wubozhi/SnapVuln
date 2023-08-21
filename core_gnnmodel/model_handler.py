import time
import torch
import torch.backends.cudnn as cudnn
from .model import Model, evaluate_predictions
from .utils.data_utils import prepare_datasets, DataStream, vectorize_input
from .utils import Timer, DummyLogger, AverageMeter


class ModelHandler_GNN(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """
    def __init__(self, config):
        # Evaluation Metrics:
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        # Prepare datasets
        datasets = prepare_datasets(config)
        self.train_set = datasets['train']
        self.dev_set = datasets['dev']

        self._n_train_examples = 0
        self.model = Model(config, self.train_set)
        self.model.network = self.model.network.to(self.device)

        self.train_loader = DataStream(self.train_set, self.model.vocab_model.word_vocab,
                                       self.model.vocab_model.edge_vocab,
                                       config=config, isShuffle=True, isLoop=True, isSort=True)
        self._n_train_batches = self.train_loader.get_num_batch()

        self.dev_loader = DataStream(self.dev_set, self.model.vocab_model.word_vocab,
                                     self.model.vocab_model.edge_vocab,
                                     config=config, isShuffle=False, isLoop=True, isSort=True)
        self._n_dev_batches = self.dev_loader.get_num_batch()

        self.config = self.model.config
        self.is_test = False

    def train(self):
        self.is_test = False
        timer = Timer("Train")
        if self.config['pretrained']:
            self._epoch = self._best_epoch = self.model.saved_epoch
        else:
            self._epoch = self._best_epoch = 0
        self._reset_metrics()
        self._best_metrics = {self.config['early_stop_metric']: 0}
        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1
            print("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self.logger.write_to_file("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            labels, probs, files = self._run_epoch(self.train_loader, training=True, verbose=self.config['verbose'])
            train_epoch_time = timer.interval("Training Epoch {}".format(self._epoch))
            format_str = "Training Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._train_loss.mean())
            train_metrics = evaluate_predictions(labels, probs)
            format_str += self.plain_metric_to_str(train_metrics)
            self.logger.write_to_file(format_str)
            print(format_str)
            print("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self.logger.write_to_file("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            labels, probs, files = self._run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'])
            timer.interval("Validation Epoch {}".format(self._epoch))
            format_str = "Validation Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._dev_loss.mean())
            dev_metrics = evaluate_predictions(labels, probs)
            format_str += self.plain_metric_to_str(dev_metrics)
            self.logger.write_to_file(format_str)
            print(format_str)
            self.model.scheduler.step(dev_metrics[self.config['early_stop_metric']])
            if self._best_metrics[self.config['early_stop_metric']] < dev_metrics[self.config['early_stop_metric']]:
                self._best_epoch = self._epoch
                for k in dev_metrics:
                    self._best_metrics[k] = dev_metrics[k]

                if self.config['save_params']:
                    self.model.save(self.dirname, self._epoch)
                    print('Saved model to {}'.format(self.dirname))
                format_str = "!!! Updated: " + self.best_metric_to_str(self._best_metrics)
                self.logger.write_to_file(format_str)
                print(format_str)
            self._reset_metrics()

        timer.finish()
        self.training_time = timer.total

        print("Finished Training: {}".format(self.dirname))
        print(self.summary())
        return self._best_metrics

    def _run_epoch(self, data_loader, training=True, verbose=10):
        start_time = time.time()
        files = []
        if training:
            mode = 'train'
        elif self.is_test:
            mode = 'test'
        else:
            mode = 'dev'
        if training:
            self.model.optimizer.zero_grad()
        for step in range(data_loader.get_num_batch()):
            input_batch = data_loader.nextBatch()
            x_batch = vectorize_input(input_batch, training=training, device=self.device, mode=mode)
            if not x_batch:
                continue  # When there are no examples in the batch
            res = self.model.predict(x_batch, step, update=training, mode=mode)
            if 'loss' in res.keys():
                loss = res['loss']
                self._update_metrics(loss, training=training)
            if training:
                self._n_train_examples += x_batch['batch_size']

            if (verbose > 0) and (step > 0) and (step % verbose == 0):
                summary_str = self.self_report(step, mode)
                self.logger.write_to_file(summary_str)
                print(summary_str)
                print('used_time: {:0.2f}s'.format(time.time() - start_time))
            if step == 0:
                labels = x_batch['targets']
                probs = res['probs']
                files.extend(x_batch['files'])
            else:
                labels = torch.cat((labels, x_batch['targets']))
                probs = torch.cat((probs, res['probs']))
                files.extend(x_batch['files'])
        return labels, probs, files

    def self_report(self, step, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_train_batches, self._train_loss.mean())
        elif mode == "dev":
            format_str = "[predict-{}] step: [{} / {}] | loss = {:0.5f}".format(
                    self._epoch, step, self._n_dev_batches, self._dev_loss.mean())
        else:
            raise ValueError('mode = {} not supported.' % mode)
        return format_str

    def plain_metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.5f}\n'.format(k.upper(), metrics[k])
        return format_str

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(self._best_epoch) + self.best_metric_to_str(self._best_metrics)
        end = "<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def _update_metrics(self, loss, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
        else:
            if loss:
                self._dev_loss.update(loss)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True