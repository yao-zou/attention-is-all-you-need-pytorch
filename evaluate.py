from __future__ import print_function, division

import torch
from torch.autograd import Variable
from torch.nn import NLLLoss
import tqdm

from transformer import Constants


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        criterion: criterion function for evaluator
    """

    def __init__(self, criterion=NLLLoss()):
        self.criterion = criterion

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

    def evaluate(self, model, data_loader):
        """ Evaluate a model_state on given dataset and return performance.

        Args:
            model (seq2seq.models): model_state to evaluate
            data_loader (torch.util.DataLoader): data loader to evaluate against

        Returns:
            criterion(float): criterion of the given model_state on the given dataset
        """
        total_words = 0
        total_correct = 0
        total_loss = 0
        print("evaluate")
        for batch in tqdm.tqdm(data_loader):
            src, tgt, gold = self._get_batch(batch)
            predictions = model(src, tgt)
            loss = self.criterion(predictions, gold.contiguous().view(-1))

            n_words = gold.data.ne(Constants.PAD).sum()
            predictions = predictions.max(1)[1]
            gold = gold.contiguous().view(-1)
            n_correct = predictions.data.eq(gold.data)
            n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()
            total_loss += loss.data[0]
            total_correct += n_correct
            total_words += n_words

        return total_loss / total_words, total_correct / total_words
