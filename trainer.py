import logging
import random

import os
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid

from evaluate import Evaluator
from transformer import Constants
from utils.checkpoint import Checkpoint
from tensorboardX import SummaryWriter


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        criterion: criterion function for training
         optimizer (seq2seq_model.optim.Optimizer): optimizer for training
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
    """

    def __init__(self, criterion, model, optimizer, expt_dir='experiment',
                 print_every=10, evaluate_every=100):
        self._trainer = "Simple Trainer"
        self.logger = logging.getLogger(__name__)
        self.criterion = criterion
        self.model = model
        self.evaluator = Evaluator(criterion=self.criterion)
        self.optimizer = optimizer
        self.print_every = print_every
        self.evaluate_every = evaluate_every
        self.writer = SummaryWriter()

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.best_accuracy = 0

    @staticmethod
    def _get_batch(batch):
        (src_seq, src_pos), (tgt_seq, tgt_pos), sentences = batch
        # prepare data
        src_seq = Variable(src_seq)
        src_pos = Variable(src_pos)
        tgt_seq = Variable(tgt_seq)
        tgt_pos = Variable(tgt_pos)
        if torch.cuda.is_available():
            src_seq = src_seq.cuda()
            src_pos = src_pos.cuda()
            tgt_seq = tgt_seq.cuda()
            tgt_pos = tgt_pos.cuda()
        src = (src_seq, src_pos)
        tgt = (tgt_seq, tgt_pos)
        gold = tgt[0][:, 1:]
        return src, tgt, gold

    def _train_batch(self, src, tgt, gold):
        self.optimizer.zero_grad()
        pred = self.model(src, tgt)
        loss = self.criterion(pred, gold.contiguous().view(-1))

        n_words = gold.data.ne(Constants.PAD).sum()
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        n_correct = pred.data.eq(gold.data)
        n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()

        loss.backward()
        self.optimizer.step()
        self.optimizer.update_learning_rate()
        return loss.data[0], n_correct, n_words

    def _train_epoch(self, n_epoch, data_loader, dev_data_loader):
        batch_iterator = data_loader.__iter__()
        total_steps = len(data_loader)
        print_loss_total = 0  # Reset every print_every
        total_correct = 0
        total_words = 0
        step = 0
        for batch in batch_iterator:
            step += 1
            self.model.train(True)
            src, tgt, gold = self._get_batch(batch)
            loss, correct, words = self._train_batch(src, tgt, gold)
            # Record average criterion
            print_loss_total += loss
            total_correct += correct
            total_words += words

            if step % self.print_every == 0 and step >= self.print_every:
                print_loss_avg = print_loss_total / total_words
                print_loss_total = 0
                accuracy = total_correct / total_words
                total_correct = 0
                total_words = 0
                log_msg = 'Progress: epoch %d,  %d / %d, Train: %.4f, accuracy %.4f' % (
                    n_epoch,
                    step, total_steps,
                    print_loss_avg,
                    accuracy)
                self.writer.add_scalars("data/losses",
                                        {"training loss": print_loss_avg}, step)
                self.writer.add_scalars("data/accuracies", {"train": accuracy}, step)
                lr = self.optimizer.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("data/learning_rate", lr, step)
                log_msg += ", lr: {}".format(lr)
                self.logger.info(log_msg)
                batch, seq, channel, height, width = src[0].size()

                image = make_grid(src[0].view(-1, channel, height, width), nrow=seq)
                self.writer.add_image("data/images", image, step)

            if step % self.evaluate_every == 0 or step == total_steps:
                self.model.eval()
                dev_loss, dev_accuracy = self.evaluator.evaluate(self.model, dev_data_loader)
                log_msg = "evaluate in epoch %d, step %d, Dev loss: %.4f, Accuracy: %.4f" \
                          % (n_epoch, step, dev_loss, dev_accuracy)
                self.model.train(mode=True)
                self.writer.add_scalars("data/losses",
                                        {"dev loss": dev_loss}, step)
                self.writer.add_scalars("data/accuracies", {"dev": dev_accuracy}, step)
                self.logger.info(log_msg)
                if dev_accuracy > self.best_accuracy:
                    Checkpoint(model_state=self.model.state_dict(),
                               optimizer_state=self.optimizer.optimizer.state_dict(),
                               optimizer_current_step=self.optimizer.n_current_steps,
                               epoch=n_epoch,
                               ).save(self.expt_dir)
                    self.best_accuracy = dev_accuracy

    def train(self, data_loader, dev_data_loader, num_epochs=5, start_epoch=0):
        """ Run training for a given model_state.

        Args:
            data_loader (torch.utils.DataLoader): dataset loader to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            start_epoch: (int, optional): the epoch to start training (default 0)
            dev_data_loader (torch.utils.DataLoader, optional): dev dataset loader (default None)
        Returns:
            model (seq2seq.models): trained model.
        """
        log = self.logger

        steps_per_epoch = len(data_loader)
        total_steps = steps_per_epoch * num_epochs

        log.info("from epoch {} to epoch {}".format(start_epoch, total_steps))
        log.info("steps per epoch is {}, total steps is {}".format(steps_per_epoch, total_steps))
        for epoch in range(start_epoch, num_epochs):
            self._train_epoch(epoch, data_loader, dev_data_loader)
        return self.model
