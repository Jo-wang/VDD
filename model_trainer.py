from numpy.core.defchararray import partition
import torch
from torch import nn
import torch.nn.functional as F
import models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os.path as osp
import os
from tqdm import tqdm
from torch.autograd import Variable

from utils.logger import AverageMeter as meter
from data_loader import Office_Dataset, Digits_Dataset, DomainNet_Dataset, PACS_Dataset, CIFAR_Dataset

from utils.loss import FocalLoss, get_loss_f, cross_entropy_soft, osbp_loss, ExLoss, AngularPenaltySMLoss
import random
from models.component import Discriminator, Classifier, Domain_Classifier
import numpy as np
from torch.nn import DataParallel

import shutil


best_H = 0

class ModelTrainer():
    def __init__(self, args, data, step=0, logger=None):
        self.args = args
        self.args.num_domains = data.num_domains
        self.batch_size = args.batch_size
        self.data_workers = 6

        self.step = step
        self.data = data

        self.num_class = data.num_class
        self.class_name = data.class_name
        self.num_domains = data.num_domains
        self.num_task = args.batch_size
        self.mean_pix = data.mean_pix
        self.std_pix = data.std_pix

        self.device_ids = [0, 1]

        self.model = models.create(args.arch, args).cuda()
        self.model = DataParallel(self.model, self.device_ids).cuda()

        self.meter = meter(args.num_class)
        c_class = self.num_class - 1 if args.open_method == 'OSVM' else self.num_class

        if args.dataset == 'digits':
            self.threshold = 1 / c_class + 0.1
        elif args.dataset == 'office':
            self.threshold = 1 / c_class + 0.8
        elif args.dataset == 'domain_net':
            self.threshold = 1 / c_class + 0.4
        elif args.dataset == 'pacs':
            self.threshold = 1 / c_class + 0.6
            # self.dataset = 'pacs'
        elif args.dataset == 'cifar':
            self.threshold = 1 / c_class + 0.2

        # CE for classification
        if args.CE_loss is 'CE':
            self.criterionCE = nn.CrossEntropyLoss(reduction='mean')
        elif args.CE_loss is 'focal':
            self.criterionCE = FocalLoss()
        elif args.CE_loss is 'momentum':
            self.criterionCE = nn.CrossEntropyLoss(reduction='mean')
            self.criterionM = ExLoss(args.in_features, self.num_class).cuda()
        elif args.CE_loss is 'angular':
            self.criterionCE =AngularPenaltySMLoss(args.in_features, self.num_class).cuda()
        self.criterionBCE = nn.BCEWithLogitsLoss()
        self.alpha = 0
        self.global_step = 0
        self.logger = logger
        self.val_acc = 0

        self.delta_known = 0.9
        self.delta_unk = 0.3
        self.rescale = 32

        if self.args.CE_loss is not 'momentum':
            self.classifier = Classifier(args, nclass=c_class)
            # self.classifier = self.classifier.cuda()
            self.classifier = DataParallel(self.classifier, self.device_ids).cuda()

        if self.args.discriminator:
            self.discriminator = Discriminator(self.args)
            # self.discriminator = self.discriminator.cuda()
            self.discriminator = DataParallel(self.discriminator, self.device_ids).cuda()

        self.reference_img = []
        self.update_flag = True

        # @note  for each class in each domain, initial a reference image as None
        for i in range(self.num_domains):
            temp = {}
            for j in range(self.num_class - 1):
                temp[j] = None
            self.reference_img.append(temp)

        # change the learning rate
        param_groups = [
            {'params': self.model.parameters(), 'lr_mult': 0.1},
            # {'params': self.model.decoder.parameters(), 'lr_mult': 0.1},
            # {'params': self.domain_classifier.parameters(), 'lr_mult': 1},
            {'params': self.classifier.parameters(), 'lr_mult': 1}
        ]
        if self.args.discriminator:
            param_groups.append({'params': self.discriminator.parameters(), 'lr_mult': 1})

        if self.args.method == "MSDA":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=args.lr,
                                              weight_decay=args.weight_decay)
            self.optimizer_c = torch.optim.Adam(params=self.classifier.parameters(),
                                              lr=args.lr,
                                              weight_decay=args.weight_decay)

        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, self.args.max_epoch)
        if args.resume:
            for root, dirs, files in os.walk("./checkpoints"):
                for name in files:
                    path = os.path.join(root, name)
            self.load_model_weight(path)

    def get_dataloader(self, dataset, training=False):

        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_workers,
                                 shuffle=training, pin_memory=True, drop_last=training)
        return data_loader

    def pseudo_target(self, inputs):

        try_outputs = F.softmax(inputs, dim=-1)
        idx_known = try_outputs[:, :-1].max(dim=-1)[0] > self.delta_known
        idx_unk = try_outputs[:, :-1].max(dim=-1)[0] < self.delta_unk
        inputs_list = []
        targets_list = []
        if idx_known.any():
            inputs_list.append(inputs[idx_known])
            targets_list.append(try_outputs[idx_known, :-1].max(dim=-1)[1].long())
        if idx_unk.any():
            inputs_list.append(inputs[idx_unk])
            targets_list.append((self.num_class - 1) * torch.ones(inputs[idx_unk].size()[0], ).long().cuda())
        # nothing to learn for target
        if len(inputs_list) == 0:
            return None, None
        else:
            inputs = torch.cat(inputs_list, dim=0)
            targets = torch.cat(targets_list, dim=0).long()
        return inputs, targets

    def transform_shape(self, tensor):

        batch_size, num_class, other_dim = tensor.shape
        tensor = tensor.contiguous().view(batch_size * num_class, other_dim)
        return tensor

    def recover_img(self, img, top=3):
        img_ori = None
        if type(img) is dict:
            for key in img.keys():
                if img[key] is not None:
                    img_ori = (img[key] * self.std_pix.cuda()) + self.mean_pix.cuda()
                    img_ori = img_ori.unsqueeze(0)
                    break
        else:
            img_ori = (img[:top, :] * self.std_pix.cuda()) + self.mean_pix.cuda()
        return img_ori

    def update_reference(self, images, labels):
        batch_size, num_domains = labels.squeeze().size()

        # @note update image reference
        # labels = labels.squeeze()
        # @todo why labels[0,2] can be 5? This make self.reference_img out of boundary
        for i in range(batch_size):
            for j in range(num_domains):
                if self.reference_img[j][int(labels[i, j])] is None:
                    # if self.dataset == 'pacs':
                    #     self.reference_img[j][int(labels[i, j]) - 1] = F.interpolate(images[i, j].unsqueeze(0), size=(self.rescale, self.rescale)).squeeze()
                    # else:
                    self.reference_img[j][int(labels[i, j])] = F.interpolate(images[i, j].unsqueeze(0), size=(self.rescale, self.rescale)).squeeze()

        # check if need further update
        state = [any(v is None for v in class_samp.values()) for class_samp in self.reference_img[:-1]]
        self.update_flag = any(v is True for v in state)


    
    def train(self, step, epochs=1):
        args = self.args
        self.data.shuffle_datasets()
        self.alpha = 0.5 / args.max_epoch * step
        train_loader = self.get_dataloader(self.data, training=True)
        # @note read at here
        self.criterionVAE = get_loss_f(args.loss, n_data=len(train_loader) * args.batch_size, **vars(args))
        # gif_visualizer = GifTraversalsTraining(self.model, args.dataset, exp_dir)
        self.model.train()
        self.classifier.train()
        # self.domain_classifier.train()
        if self.args.discriminator:
            self.discriminator.train()
        self.meter.reset()

        for epoch in range(epochs):
            with tqdm(total=len(train_loader)) as pbar:
                # @todo the doamin number of inouts is not correct
                for i, inputs in enumerate(train_loader):
                    outputs = []
                    loss = 0
                    loss_adv = 0
                    images = Variable(inputs[0], requires_grad=False).cuda()
                    labels = Variable(inputs[1]).cuda().unsqueeze(-1)
                    domain_labels = Variable(inputs[2]).cuda()
                    if self.update_flag and step < 2:
                        self.update_reference(images[:, :-1], labels[:, :-1]) 

                    # extract backbone features
                    for domain_id in range(images.size()[1]):
                        output = self.model(images[:, domain_id, :, :, :], domain_labels, domain_id)
                        # with SummaryWriter(comment='VAE') as w:
                        #     w.add_graph(self.model, (images[:, domain_id, :, :, :], domain_labels, torch.from_numpy（np.array(domain_id）)))

                        self.logger.log_images('original/{}_domain'.format(domain_id),  self.recover_img(images[:, domain_id, :]),
                                                   self.logger.global_step)
                        self.logger.log_images('generated/{}_domain'.format(domain_id), self.recover_img(output[0]),
                                                   self.logger.global_step)
                        # only for debug

                        self.logger.log_images('debug/{}_domain'.format(domain_id),
                                               self.recover_img(output[5]), self.logger.global_step)

                        if self.args.discriminator:
                            coef = 1 / (self.num_domains - 1) if (domain_id < self.num_domains - 1) else 1
                            disc_logits = self.discriminator(output[4])
                            domain_label = torch.tensor(domain_id < self.num_domains - 1).double() * torch.ones(disc_logits.size()[0]).cuda()
                            loss_adv += coef * self.criterionBCE(disc_logits, domain_label)

                        outputs.append(output)

                    all_recon = torch.stack([item[0] for item in outputs], dim=1)
                    all_d_dist = (torch.stack([item[1][0] for item in outputs], dim=1),
                                  torch.stack([item[1][1] for item in outputs], dim=1))
                    all_d_samp = torch.stack([item[2] for item in outputs], dim=1)
                    all_z_dist = (torch.stack([item[3][0] for item in outputs], dim=1),
                                  torch.stack([item[3][1] for item in outputs], dim=1))
                    all_z_samp = torch.stack([item[4] for item in outputs], dim=1)

                    # reconstruction and disentaglement

                    all_fake_recon = torch.stack([item[5] for item in outputs], dim=1)

                    all_logits = self.classifier(all_z_samp)  # @note 从encoder_z拿到的结果直接做分类
                    # with SummaryWriter(comment='classifier') as w:
                    #         w.add_graph(self.classifier, (all_z_samp,))

                    loss_source = self.criterionCE(self.transform_shape(all_logits[:, :-1, :]),
                                                   labels[:, :-1, :].contiguous().view(-1))

                    loss_target = cross_entropy_soft(all_logits[:, -1, :])

                    value, indices = torch.softmax(all_logits[:, -1, :].detach(), dim=-1).max(dim=-1)
                    estimated_labels = torch.cat((labels[:, :-1, :], indices.unsqueeze(-1).unsqueeze(-1)), dim=1)

                    loss_diverse = 0

                    for domain_id in range(images.size()[1]):
                        fake_domain_id = outputs[domain_id][6]
                        for j in range(fake_domain_id[0]):
                            # temp_class = int(estimated_labels[j, domain_id])
                            # if temp_class < self.num_class - 1:
                            #     if self.reference_img[int(fake_domain_id[j])][temp_class] is not None:
                            #         loss_diverse += F.mse_loss(255 * outputs[domain_id][5][j], 255 * self.reference_img[int(fake_domain_id[j])][temp_class])
                            hit = torch.where(estimated_labels[:, fake_domain_id[j]] == estimated_labels[j, domain_id])[0]
                            if len(hit) > 0:
                                # @note estimated_labels 16*5*1 for Digits
                                # @note output 7*16*3*32*32 for Digits
                                loss_diverse += F.mse_loss(255 * outputs[domain_id][5][j], 255 * F.interpolate(images[hit[0], fake_domain_id[j]].unsqueeze(0), 
                                                size=(self.rescale, self.rescale), mode='bilinear').squeeze(), reduction='sum') / 1000

                    if self.args.open_method == 'Pseudo':
                        new_targets, new_labels = self.pseudo_target(all_logits[:, -1, :])
                        if new_targets is not None:
                            loss_target_pseudo = self.criterionCE(new_targets, new_labels)
                            loss += loss_target_pseudo
                    prob_tar = F.softmax(all_logits[:, -1, :], 1)
                    if args.open_method == 'OSBP':
                        prob_tar_known = torch.sum(prob_tar[:, :-1], 1).view(-1, 1)
                        prob_tar_unk = prob_tar[:, -1].contiguous().view(-1, 1)
                        target_funk = torch.cat((torch.FloatTensor(images.size()[0], 1).fill_(0.5), #know class/ unk 5/7:2/7
                                                 torch.FloatTensor(images.size()[0], 1).fill_(0.5)), dim=1).cuda()
                        loss_osbp = osbp_loss(torch.cat((prob_tar_known, prob_tar_unk), 1), target_funk)
                        loss += loss_osbp

                    loss_f = self.criterionVAE(images, all_recon, all_d_dist, all_d_samp, all_z_dist, all_z_samp)

                    # loss = loss_source * 2 + 2 * loss_osbp + loss_target + loss_f + (1 + self.alpha) * loss_diverse

                    loss += loss_source * 2 + loss_target + loss_f + self.alpha * loss_diverse
                    # 调整 alpha loss——f加weight

                    if self.args.discriminator:
                         loss += loss_adv
                    # for debug only: update target class accuracy
                    # @note target class has no label 5: unk
                    self.meter.update(labels[:, -1, :].detach().cpu().view(-1).numpy(),
                                      torch.argmax(prob_tar, -1).eq(labels[:, -1, :].squeeze()).double().detach().cpu().numpy())
                    # if epoch == 40:
                    #     a = self.meter
                    del outputs
                    self.optimizer.zero_grad()
                    self.optimizer_c.zero_grad()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer_c.step()
                    # self.lr_scheduler.step()

                    self.logger.global_step += 1
                    self.criterionVAE.n_train_steps += 1
                    if self.args.discriminator:
                        self.logger.log_scalar('train/adv_loss', loss_adv, self.logger.global_step)
                    self.logger.log_scalar('train/source_loss', loss_source, self.logger.global_step)
                    self.logger.log_scalar('train/diverse_loss', loss_diverse, self.logger.global_step)
                    # self.logger.log_scalar('train/domain_loss', loss_domain, self.logger.global_step)
                    # self.logger.log_scalar('train/tar_pseudo_loss', loss_target_pseudo, self.logger.global_step)
                    # self.logger.log_scalar('train/target_loss', loss_target, self.logger.global_step)
                    # self.logger.log_scalar('train/osbp_loss', loss_osbp, self.logger.global_step)
                    self.logger.log_scalar('train/vae_loss', loss_f, self.logger.global_step)
                    self.logger.log_scalar('train/OS_star', self.meter.avg[:-1].mean(), self.logger.global_step)
                    self.logger.log_scalar('train/OS', self.meter.avg.mean(), self.logger.global_step)
                    self.logger.log_scalar('train/ALL', self.meter.sum.sum() / self.meter.count.sum(), self.logger.global_step)
                    self.logger.log_scalar('train/UNK', self.meter.avg[-1], self.logger.global_step)
                    pbar.update()

                self.meter.reset()

        # # save model
        # states = {'model': self.model.state_dict(),
        #           'classifier': self.classifier.state_dict(),
        #           'model-module': self.model.module.state_dict(),
        #           'classifier-module': self.classifier.module.state_dict(),
        #           'iteration': self.logger.global_step,
        #           'optimizer': self.optimizer.state_dict()}
        # torch.save(states, osp.join(args.checkpoints_dir, '{}_step_{}.pth.tar'.format(args.experiment, step)))
        self.meter.reset()


    def estimate_label(self, step):
        global best_H
        self.step = step
        args = self.args
        print('target evaluation...')
        if args.dataset == 'domain_net':
            test_data = DomainNet_Dataset(args=args, partition='test')
        elif args.dataset == 'office':
            test_data = Office_Dataset(args=args, partition='test')
        elif args.dataset == 'digits':
            test_data = Digits_Dataset(args=args, partition='test')
        elif args.dataset == 'pacs':
            test_data = PACS_Dataset(args=args, partition='test')
        elif args.dataset == 'cifar':
            test_data = CIFAR_Dataset(args=args, partition='test')

        self.meter.reset()
        # append labels and scores for target samples

        target_loader = self.get_dataloader(test_data, training=False)
        self.model.eval()
        # self.domain_classifier.eval()
        self.classifier.eval()
        if self.args.discriminator:
            self.discriminator.eval()
        labels_list = []
        # (recon, d_dist, d_samp, z_dist, z_samp, fake_recon, fake_d_samp)
        with tqdm(total=len(target_loader)) as pbar:
            for i, (images, labels, domain_labels) in enumerate(target_loader):

                images = Variable(images, requires_grad=False).cuda()
                labels = Variable(labels).cuda()
                domain_labels = domain_labels.cuda()

                outputs = self.model(images[:, -1, :, :, :], domain_labels, 0)
                labels_list = labels_list + list((labels.detach().cpu().numpy() < self.num_class - 1).astype(int))

                # feat = torch.cat((outputs[2], outputs[4]), dim=1)
                feat = outputs[4]     
                logits = self.classifier(feat.unsqueeze(1)).squeeze()
                prob_tar = F.softmax(logits, -1)
                # only for debugging
                target_labels = labels[:, -1].view(-1)

                pred = torch.argmax(prob_tar, -1)
                if self.args.open_method == 'OSVM':
                    idx = prob_tar[:, :-1].max(-1)[0].detach() < self.threshold
                    pred[idx] = self.num_class - 1  # unk class

                target_prec = pred.eq(target_labels).detach().cpu().double()

                self.meter.update(
                    target_labels.detach().cpu().view(-1).data.cpu().numpy(),
                    target_prec.numpy())

                pbar.update()

        print('Step: {} | {}; \t'
              'OS Prec {:.4%}\t'
              'OS* Prec {:.4%}\t'
              'ALL Prec {:.4%}\t'
              'UNK Prec {:.4%}\t'
              .format(i, len(target_loader),
                      self.meter.avg.mean(),
                      self.meter.avg[:-1].mean(),
                      self.meter.sum.sum() / self.meter.count.sum(),
                      self.meter.avg[-1],
                      ))

        labels = np.array(labels_list)

        u, counts = np.unique(labels, return_counts=True)
        known_ratio = counts[0] / len(labels)
        unk_ratio = counts[1] / len(labels)
        ac = self.meter.sum[:-1].sum() / self.meter.count[:-1].sum()
        ac_hat = self.meter.avg[-1]
        if ac_hat != 0.0:
            print(ac_hat)
        H_score = 2 * (ac * ac_hat) / (ac + ac_hat)
        new_balanced = 2 * (self.meter.avg[:-1].mean() * ac_hat) / (self.meter.avg[:-1].mean() + ac_hat)

        for k in range(self.num_class):
            self.logger.log_scalar('test/{}_class'.format(k), self.meter.avg[k], self.step)
        self.logger.log_scalar('test/ALL', self.meter.sum.sum() / self.meter.count.sum(), self.step)
        self.logger.log_scalar('test/OS*', self.meter.avg[:-1].mean(), self.step)
        self.logger.log_scalar('test/OS', self.meter.avg.mean(), self.step)
        self.logger.log_scalar('test/UNK', self.meter.avg[-1], self.step)
        self.logger.log_scalar('test/new_balanced', new_balanced, self.step)
        self.logger.log_scalar('test/H_score', H_score, self.step)
        if H_score > best_H:
            best_H = H_score
            states = {
                  'model-module': self.model.module.state_dict(),
                  'classifier-module': self.classifier.module.state_dict(),
                #   'iteration': self.logger.global_step,
                  'optimizer': self.optimizer.state_dict(),
                  'optimizer-c' : self.optimizer_c.state_dict()}
            torch.save(states, osp.join('/home/s4565257/MSOUDA_ICME/checkpoints', '{}_best.pth.tar'.format(args.experiment)))

        self.meter.reset()
        self.model.train()
        # self.domain_classifier.eval()
        self.classifier.train()


        # return pred_labels.data.cpu().numpy(), pred_scores.data.cpu().numpy(), real_labels.data.cpu().numpy()


    # def save_checkpoint(self, state, is_best, path_exp, filename='checkpoint.pth.tar'):
    #     path_file = path_exp + filename
    #     torch.save(state, path_file)
        
	#     if is_best:
    # 		    path_best = path_exp + 'model_best.pth.tar'
	# 	    shutil.copyfile(path_file, path_best)


    def extract_feature(self):
        print('Feature extracting...')
        self.meter.reset()
        # append labels and scores for target samples
        vgg_features_target = []
        node_features_target = []
        labels = []
        overall_split = []
        target_loader = self.get_dataloader(self.data, training=False)
        self.model.eval()
        self.gnnModel.eval()
        num_correct = 0
        skip_flag = self.args.visualization
        with tqdm(total=len(target_loader)) as pbar:
            for i, (images, targets, target_labels, _, split) in enumerate(target_loader):

                # for debugging
                # if i > 100:
                #     break
                images = Variable(images, requires_grad=False).cuda()
                targets = Variable(targets).cuda()

                # only for debugging
                # target_labels = Variable(target_labels).cuda()

                targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)
                target_labels = self.transform_shape(target_labels.unsqueeze(-1)).squeeze(-1).cuda()
                init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(
                    targets)
                # gt_edge = self.label2edge_gt(target_labels)
                # extract backbone features
                features = self.model(images)
                features = self.transform_shape(features)

                # feed into graph networks
                edge_logits, node_feat = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge,
                                                       target_mask=target_edge_mask)
                vgg_features_target.append(features.data.cpu())
                #####heat map only
                # temp = np.array(edge_logits[0].data.cpu()) * 4
                # ax = sns.heatmap(temp.squeeze(), vmax=1)#
                # cbar = ax.collections[0].colorbar
                # # here set the labelsize by 20
                # cbar.ax.tick_params(labelsize=17)
                # plt.savefig('heat/' + str(i) + '.png')
                # plt.close()
                ###########
                node_features_target.append(node_feat[-1].data.cpu())
                labels.append(target_labels.data.cpu())
                overall_split.append(split)
                if skip_flag and i > 50:
                    break

                pbar.update()

        return vgg_features_target, node_features_target, labels, overall_split
