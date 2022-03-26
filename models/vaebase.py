from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from .component import grad_reverse
from torch import logsumexp

import sys 
sys.path.append("..") 



__all__ = ["VAEBase"]

"""
def init_specific_model(args):

    model = VAEBase(args)
    # model.model_type = model_type  # store to help reloading
    return model
"""

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DomainEncoder(nn.Module):
    def __init__(self, num_domains, output_dim):
        super(DomainEncoder, self).__init__()
        self.latent_dim = output_dim
        self.embed = nn.Embedding(num_domains, 512)
        self.bn = nn.BatchNorm1d(512)
        self.mu_logvar_gen = nn.Linear(512, output_dim * 2)

        # setup the non-linearity
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        h = self.act(self.bn(self.embed(inputs)))
        mu_logvar = self.mu_logvar_gen(h)
        outputs = mu_logvar.view(batch_size, self.latent_dim, 2).unbind(-1)

        return outputs

class ConvEncoder(nn.Module):
    def __init__(self, output_dim):  # latent output dimensions
        super(ConvEncoder, self).__init__()
        self.latent_dim = output_dim
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.mu_logvar_gen = nn.Linear(2048, output_dim * 2)

        # setup the non-linearity
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        assert len(inputs.shape) == 4
        batch_size, channel, width, height = inputs.size()
        h = inputs.contiguous().view(-1, channel, width, height)
        h = F.max_pool2d(self.act(self.bn1(self.conv1(h))),  stride=2, kernel_size=3, padding=1)
        h = F.max_pool2d(self.act(self.bn2(self.conv2(h))), stride=2, kernel_size=3, padding=1)
        h = self.act(self.bn3(self.conv3(h)))
        # [CHECK] did not add dropout so far
        h = h.view(batch_size, -1)
        h = self.act(self.bn1_fc(self.fc1(h)))
        h = self.act(self.bn2_fc(self.fc2(h)))
        mu_logvar = self.mu_logvar_gen(h)

        outputs = mu_logvar.view(batch_size, self.latent_dim, 2).unbind(-1)

        return outputs

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class ResDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ResDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, 3 * 32 ** 2)
        self.l1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True))
        resblocks = []
        for _ in range(2):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)
        #@note change 64 to 32
        self.l2 = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, z):
        # h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.fc(z).view(z.size(0), 3, 32, 32)
        out = self.l1(h)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 2, 1)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 3, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = torch.sigmoid(self.conv_final(h))
        return mu_img

class VAEBase(nn.Module):
    def __init__(self, args):
        super(VAEBase, self).__init__()
        self.args = args
        self.domain_list = np.arange(args.num_domains - 1)
        # create the encoder and decoder networks
        self.encoder_d = DomainEncoder(args.num_domains, args.domain_in_features)

        self.encoder_z = ConvEncoder(args.in_features)
        # self.decoder = ConvDecoder(args.in_features + args.domain_in_features)
        
        self.decoder = ResDecoder(args.in_features + args.domain_in_features)


    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x, domain_labels, domain_id):
        """
        Forward pass of model.
        Parameters
        ----------checkpoint
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        fake_domains = np.delete(self.domain_list, domain_id) if domain_id < self.args.num_domains - 1 else self.domain_list
        fake_domain_labels = torch.tensor(random.choices(fake_domains, k=domain_labels.size()[0])).cuda()
        d_latent_dist = self.encoder_d(domain_labels[:, domain_id])
        d_fake_latent_dist = self.encoder_d(fake_domain_labels)
        z_latent_dist = self.encoder_z(x)
        d_latent_sample = self.reparameterize(*d_latent_dist)
        d_fake_latent_sample = self.reparameterize(*d_fake_latent_dist)
        z_latent_sample = self.reparameterize(*z_latent_dist)
        reconstruct = self.decoder(torch.cat((d_latent_sample, z_latent_sample), dim=1))
        fake_reconstruct = self.decoder(torch.cat((d_fake_latent_sample, z_latent_sample), dim=1))     
        return reconstruct, d_latent_dist, d_latent_sample, z_latent_dist, z_latent_sample, fake_reconstruct, fake_domain_labels

