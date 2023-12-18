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


from train import main

import argparse


if __name__ == '__main__':
    # OK Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=int, required=True)
    parser.add_argument('--displacement', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--multi_object', action='store_true')
    parser.add_argument('--prefix', type=str, default=None)
    args = parser.parse_args()

    ## Multiprocessing seems to hang randomly. We'll disable because we only use one gpu anyway.
    # torch.multiprocessing.set_start_method('spawn') 

    # Default params with commandline input
    opt = opts()
    opt = opt.parser.parse_args(args=[])

    # Local configuration
    opt.c = 'pallet'
    opt.arch='dlav1_34'
    opt.obj_scale = args.scale
    opt.obj_scale_weight = 1. if args.scale else 0.
    opt.hp_weight = 1. if args.displacement else 0.
    opt.mug = False

    # Set custom data dir. data_dir is where outf and outf_all folders are located
    opt.data_dir = f'/media/localadmin/0c21d63f-0916-4325-a37c-33263ee1cba7/home/olli/Data/pallet_carla_2{args.dataset}/'
    # opt.data_name = 'pallet' 
    # folders should be in data_dir and named `<data_name>_train/` and `<data_name>_test/`

    opt.max_objs = 10 if args.multi_object else 1

    ##
    ## For pallet training. lr step should be '20'
    # Training param
    # opt.exp_id = f'objectron_{opt.c}_{opt.arch}'
    opt.exp_id  = f"pallet_final3_ds_{args.dataset}"
    if args.prefix is not None:
        opt.exp_id += f"_{args.prefix}"
    opt.exp_id += f"_{'hp' if args.displacement else 'nohp'}"
    opt.exp_id += f"_{'scale' if args.scale else 'noscale'}"
    opt.exp_id += f"_{'multi' if args.multi_object else 'single'}"
    opt.max_load = 8000
    opt.max_load_val = 400
    opt.num_iters = 100
    opt.num_epochs = 100
    opt.val_intervals = 5
    opt.lr_step = '50'
    opt.batch_size = 16
    opt.lr = 6e-5
    opt.gpus = '0'
    opt.num_workers = 4
    opt.print_iter = 0
    opt.debug = 5
    opt.save_all = False

    # # To continue
    # opt.resume = True
    # opt.load_model = ""

    # Copy from parse function from opts.py
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1:  # init default head_conv
        opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
        opt.val_intervals = 100000000

    if opt.master_batch_size == -1:
        opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
        slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
        if i < rest_batch_size % (len(opt.gpus) - 1):
            slave_chunk_size += 1
        opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    opt.save_dir = os.path.join(opt.exp_dir, f'{opt.exp_id}')
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)

    main(opt)
