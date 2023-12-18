# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import collate_fn_filtered
from lib.trains.train_factory import train_factory
import time
import numpy as np

from lib.datasets.dataset_combined import ObjectPoseDataset


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    Dataset = ObjectPoseDataset

    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_dataset = Dataset(opt, 'val')

    if opt.tracking_task == True:
        val_dataset_subset = torch.utils.data.Subset(val_dataset, range(0, len(val_dataset), 15))
    else:
        val_dataset_subset = val_dataset

    val_loader = torch.utils.data.DataLoader(
        val_dataset_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_filtered
    )

    if opt.test:
        _, preds, _ = trainer.val(0, val_loader)

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_filtered
    )

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _, log_imgs = trainer.train(epoch, train_loader)
        logger.write('epoch: {} | '.format(epoch))  # txt logging
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)  # tensorboard logging
            logger.write('train_{} {:8f} | '.format(k, v))  # txt logging
        logger.img_summary('train', log_imgs, epoch)
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, f'{opt.c}_{mark}.pth'),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds, log_imgs = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('val_{} {:8f} | '.format(k, v))
            logger.img_summary('val', log_imgs, epoch)
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, f'{opt.c}_best.pth'),
                           epoch, model)
            # else:
            save_model(os.path.join(opt.save_dir, f'{opt.c}_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n\n')

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, f'{opt.c}_{epoch}.pth'),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


