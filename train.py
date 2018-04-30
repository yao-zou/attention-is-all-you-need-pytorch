'''
This script handling the training process.
'''
import logging
import os
from utils.dataset import LipReadingDataSet, collate_fn
from utils.checkpoint import Checkpoint
from utils.transforms import Crop
import argparse
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from torch.utils.data import DataLoader

from trainer import SupervisedTrainer


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    # parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--pretrained', action='store', dest='pre_trained', default=None,
                        help='Path to pretrained model')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--log-level', dest='log_level',
                        default='info',
                        help='Logging level.')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    opt.n_src_max_seq = 200
    opt.n_tgt_max_seq = 100

    # ========= Preparing DataLoader =========#
    train_transforms = Crop(100, 30, 30)
    val_transforms = Crop(100, 30, 30)
    train_data_set = LipReadingDataSet('/home/disk2/zouyao/data/mvlrs/mvlrs_v1/pretrain_split_train_small.csv',
                                       transforms=train_transforms)
    training_data = DataLoader(train_data_set, batch_size=opt.batch_size, shuffle=True,
                               collate_fn=collate_fn)
    val_data_set = LipReadingDataSet('/home/disk2/zouyao/data/mvlrs/mvlrs_v1/pretrain_split_val.csv',
                                     transforms=val_transforms)
    validation_data = DataLoader(val_data_set, batch_size=opt.batch_size, shuffle=True,
                                 collate_fn=collate_fn)



    opt.tgt_vocab_size = len(Constants.VOCABULARY)

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
    logging.info(opt)

    transformer = Transformer(
        opt.tgt_vocab_size,
        opt.n_src_max_seq,
        opt.n_tgt_max_seq,
        proj_share_weight=opt.proj_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    base_optimizer = optim.Adam(
            transformer.get_trainable_parameters(),
            betas=(0.9, 0.98), eps=1e-09, lr=1e-5)
    optimizer = ScheduledOptim(base_optimizer, opt.d_model, opt.n_warmup_steps)

    if opt.load_checkpoint is not None or opt.resume:
        if opt.load_checkpoint is not None:
            checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
        else:
            checkpoint_path = Checkpoint.get_latest_checkpoint(opt.expt_dir)
        logging.info("loading checkpoint from {}".format(checkpoint_path))
        resume_checkpoint = Checkpoint.load(checkpoint_path)
        transformer.load_state_dict(resume_checkpoint.model_state)
        optimizer.optimizer.load_state_dict(resume_checkpoint.optimizer_state)
        optimizer.n_current_steps = resume_checkpoint.optimizer_current_step
        start_epoch = resume_checkpoint.epoch
    else:
        start_epoch = 0

    criterion = nn.CrossEntropyLoss(ignore_index=Constants.PAD, size_average=False)

    if opt.cuda:
        transformer = transformer.cuda()
        criterion = criterion.cuda()
    supervised_trainer = SupervisedTrainer(criterion, transformer, optimizer)

    supervised_trainer.train(training_data, validation_data, num_epochs=opt.epoch, start_epoch = start_epoch)


if __name__ == '__main__':
    main()
