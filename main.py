from __future__ import print_function, absolute_import
import os

from numpy.core.fromnumeric import partition
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# torch-related packages

import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.visualization import visualize_TSNE
from torch.utils.data import DataLoader, dataloader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# CUDA_VISIBLE_DEVICES = 0,1

# data
from data_loader import Digits_Dataset, Office_Dataset, DomainNet_Dataset, PACS_Dataset, CIFAR_Dataset
from model_trainer import ModelTrainer
from utils.logger import Logger


def main(args):
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    # prepare checkpoints and log folders
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # initialize dataset
    if args.dataset == 'digits':
        args.data_dir = os.path.join(args.data_dir, 'Digits')
        data = Digits_Dataset(args=args, partition='train')
    elif args.dataset == 'office':
        args.data_dir = os.path.join(args.data_dir, 'Office')
        data = Office_Dataset(args=args, partition='train')
    elif args.dataset == 'domain_net':
        args.data_dir = os.path.join(args.data_dir, 'DomainNet')
        data = DomainNet_Dataset(root=args.data_dir, partition='train')
    elif args.dataset == 'pacs':
        args.data_dir = os.path.join(args.data_dir, 'PACS')
        data = PACS_Dataset(args=args, partition='train')
        # @todo [16, 3, 3, 224, 224] -> [batch_size, img_label, domain_label, (img_column, img_row)]
    elif args.dataset == 'cifar':
        args.data_dir = os.path.join(args.data_dir, 'CIFAR-10-C')
        data = CIFAR_Dataset(args=args, partition='train')
    
    args.class_name = data.class_name
    args.num_class = data.num_class
    # setting experiment name
    args.experiment = set_exp_name(args)
    logger = Logger(args)
    trainer = ModelTrainer(args=args, data=data, logger=logger)
    for step in range(args.max_epoch):
        # train the model
        trainer.train(step)
        trainer.estimate_label(step)



def set_exp_name(args):
    exp_name = 'D-{}'.format(args.dataset)
    exp_name += '_tar-{}'.format(args.target_domain)
    exp_name += '_A-{}'.format(args.arch)
    # exp_name += '_L-{}'.format(args.num_layers)
    exp_name += '_B-{}'.format(args.batch_size)
    exp_name += '_O-{}'.format(args.open_method)
    return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-source Open-set Domain Adaptation')
    # set up dataset & backbone embedding
    dataset = 'digits'
    if dataset == 'digits':
        unk_class = 5
        max_epoch = 120
        arch = 'vae'
    elif dataset == 'office':
        unk_class = 10
        max_epoch = 360
        arch = 'bigvae'

    elif dataset == 'domain_net':
        unk_class = 246
        max_epoch = 10000
        arch = 'bigvae'

    elif dataset == 'pacs':
        unk_class = '5'
        max_epoch = 300
        arch = 'bigvae'

    elif dataset == 'cifar':
        unk_class = 5
        max_epoch = 200
        arch = 'vae'
        

    parser.add_argument('--dataset', type=str, default=dataset, choices=['digits', 'office', 'domain_net', 'pacs', 'cifar'])
    parser.add_argument('-a', '--arch', type=str, default=arch, choices=['vae', 'bigvae'])
    parser.add_argument('--discriminator', type=bool, default=False)
    parser.add_argument('--loss', type=str, default='VAE', choices=["VAE", "betaH", "betaB", "factor", "btcvae"])
    parser.add_argument('--CE_loss', type=str, default='CE', choices=['angular', 'focal', 'CE', 'momentum'])
    parser.add_argument('--dist', type=str, default='normal', choices=['normal', 'laplace', 'flow'])
    parser.add_argument('--method', type=str, default='MSDA')
    parser.add_argument('--unk_class', type=int, default=unk_class, help="after which classes will be unknown")
    parser.add_argument('--strict_setting', type=bool, default=True)
    parser.add_argument('--open_method', type=str, default='Pseudo', choices=['OSBP', 'OSVM', 'Pseudo'])

    if dataset == 'digits':
        parser.add_argument('--target_domain', type=str, default='syn',
                            choices=['mnistm', 'mnist', 'usps', 'svhn', 'syn'])
    elif dataset == 'office':
        parser.add_argument('--target_domain', type=str, default='webcam', metavar='N',
                            choices=['amazon', 'dslr', 'webcam'], help='target domain dataset')
    elif dataset == 'domain_net':
        parser.add_argument('--target_domain', type=str, default='clipart', metavar='N',
                            choices=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                            help='target domain dataset')
    elif dataset == 'pacs':
        parser.add_argument('--target_domain', type=str, default='photo', metavar='N',
                            choices=['art_painting', 'cartoon', 'photo', 'sketch'], help='target domain dataset')
    elif dataset == 'cifar':
        parser.add_argument('--target_domain', type=str, default='contrast', metavar='N',
                            choices=['brightness', 'contrast', 'fog', 'defocus_blur', 'frost'], help='target domain dataset')

    # set up path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'data/'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'logs/'))
    parser.add_argument('--checkpoints_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'checkpoints/'))

    # verbose setting
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--max_epoch', type=int, default=max_epoch, metavar='N', help='how many epochs')
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--log_epoch', type=int, default=1)
    parser.add_argument('--eval_log_step', type=int, default=1)
    # @note 1500
    parser.add_argument('--test_interval', type=int, default=1500)

    # hyper-parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('-b', '--batch_size', type=int, default=20)
    # parser.add_argument('--threshold', type=float, default=0.1)

    parser.add_argument('--dropout', type=float, default=0.7)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # reconstruction specific
    parser.add_argument('--rec_dist', type=str, default='gaussian', choices=['bernoulli', 'gaussian', 'laplace'])
    parser.add_argument('--steps_anneal', type=int, default=1000, help='Number of annealing steps where gradually adding '
                                                                      'the regularization')
    parser.add_argument('--reg_anneal', type=int, default=1000)
    # loss-specific
    parser.add_argument('--btcvae-A', type=float,
                        default=1.,
                        help="Weight of the MI term (alpha in the paper).")
    parser.add_argument('--btcvae-G', type=float,
                        default=1.,
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    parser.add_argument('--btcvae-B', type=float,
                        default=6.,
                        help="Weight of the TC term (beta in the paper).")

    parser.add_argument('--in_features', type=int, default=2048)
    parser.add_argument('--domain_in_features', type=int, default=100) ## vae中间层

    main(parser.parse_args())
    