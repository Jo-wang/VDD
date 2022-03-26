from data_loader import CIFAR_Dataset
import torch
from torch.utils.data import DataLoader
from models.component import Discriminator, Classifier, Domain_Classifier
import models
import os
import argparse
from models.vaebase import DomainEncoder, ConvEncoder, ResDecoder

from torch.nn import DataParallel
keys_z = [
    'encoder_z.conv1.weight', 'encoder_z.conv1.bias', 'encoder_z.bn1.weight', 'encoder_z.bn1.bias', 'encoder_z.bn1.running_mean', 'encoder_z.bn1.running_var', 
    'encoder_z.bn1.num_batches_tracked', 'encoder_z.conv2.weight', 'encoder_z.conv2.bias', 'encoder_z.bn2.weight', 'encoder_z.bn2.bias', 
    'encoder_z.bn2.running_mean', 'encoder_z.bn2.running_var', 'encoder_z.bn2.num_batches_tracked', 'encoder_z.conv3.weight', 'encoder_z.conv3.bias', 
    'encoder_z.bn3.weight', 'encoder_z.bn3.bias', 'encoder_z.bn3.running_mean', 'encoder_z.bn3.running_var', 'encoder_z.bn3.num_batches_tracked', 'encoder_z.fc1.weight', 
    'encoder_z.fc1.bias', 'encoder_z.bn1_fc.weight', 'encoder_z.bn1_fc.bias', 'encoder_z.bn1_fc.running_mean', 'encoder_z.bn1_fc.running_var', 'encoder_z.bn1_fc.num_batches_tracked', 
    'encoder_z.fc2.weight', 'encoder_z.fc2.bias', 'encoder_z.bn2_fc.weight', 'encoder_z.bn2_fc.bias', 'encoder_z.bn2_fc.running_mean', 'encoder_z.bn2_fc.running_var', 
    'encoder_z.bn2_fc.num_batches_tracked', 'encoder_z.mu_logvar_gen.weight', 'encoder_z.mu_logvar_gen.bias']
keys_d = ['encoder_d.embed.weight', 'encoder_d.bn.weight', 'encoder_d.bn.bias', 'encoder_d.bn.running_mean', 
    'encoder_d.bn.running_var', 'encoder_d.bn.num_batches_tracked', 'encoder_d.mu_logvar_gen.weight', 'encoder_d.mu_logvar_gen.bias']

def main(args):
    device_ids = [0, 1]

    test_data = CIFAR_Dataset(args=args, partition='test')
    dataloader = DataLoader(test_data, batch_size=20, num_workers=6,
                                 shuffle=False, pin_memory=True, drop_last=False)
    checkpoint = torch.load("checkpoints/Threshold_0.6_D-cifar_tar-fog_A-vae_B-20_O-Pseudo_best.pth.tar")
    model = models.create(args.arch, args)
    model = DataParallel(model, device_ids).cuda()
    # model.load_state_dict(checkpoint)
    model_dict = model.state_dict()
    model_vae = checkpoint['model-module']

def load_checkpoint(checkpoint, model, opt=None):
    """
    load saved model accordingly

    """
    if not os.path.exists(checkpoint):
        raise("File does not exists {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model-module'])

    if opt:
        opt.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def get_encoder(type, in_feature, num_domains, domain_in_feature):
    """
    load model with a specific type of encoder

    """
    if type == 'encoder_z': 
        encoder = ConvEncoder(in_feature)
    else:
        encoder = DomainEncoder(num_domains, domain_in_feature)
    return encoder


def get_decoder(latent_vector):
    """
    load decoder

    """
    return ResDecoder(latent_vector)


def load_separately_from_model(keys_z, keys_d, keys_decoder, trained_model):
    pass
    
        
    

# def load_decoder():
    # pass




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
        max_epoch = 100
        arch = 'vae'
        

    parser.add_argument('--dataset', type=str, default=dataset, choices=['digits', 'office', 'domain_net', 'pacs', 'cifar'])
    parser.add_argument('-a', '--arch1', type=str, default='encoder_d', choices=['vae', 'bigvae', 'encoder_d', 'encoder_z', 'decoder'])
    parser.add_argument('-a', '--arch2', type=str, default='encoder_z', choices=['vae', 'bigvae', 'encoder_d', 'encoder_z', 'decoder'])

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
        parser.add_argument('--target_domain', type=str, default='fog', metavar='N',
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