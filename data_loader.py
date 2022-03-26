from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import random
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

#resize功能
from scipy import misc

# pytorch
import torch 
import torch.nn as nn
import torchvision

from datasets.svhn import load_svhn
from datasets.mnist import load_mnist
from datasets.mnist_m import load_mnistm
from datasets.usps_ import load_usps
from datasets.gtsrb import load_gtsrb
from datasets.synth_number import load_syn
from datasets.office import load_office
from datasets.domainnet import load_domainnet
from datasets.pacs import load_pacs
from datasets.cifar import load_cifar

class Multi_Source_Base_Dataset(data.Dataset):
    def __init__(self, root, partition):
        super(Multi_Source_Base_Dataset, self).__init__()
        # set dataset info
        self.root = root
        self.partition = partition

        self.mean_pix = [0.485, 0.456, 0.406]
        self.std_pix = [0.229, 0.224, 0.225]
        self.scale = 256
        self.crop_scale = 224
        normalize = transforms.Normalize(mean=self.mean_pix, std=self.std_pix)

        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(self.crop_scale),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   transforms.CenterCrop(self.crop_scale),
                                                   transforms.ToTensor(),
                                                   normalize])
    def __len__(self):

        if self.partition == 'train':
            return int(min(sum(self.alpha), len(self.target_image)) / (self.num_class - 1))
        elif self.partition == 'test':
            return int(len(self.target_image) / (self.num_class - 1))

    def __getitem__(self, item):

        image_data = []
        label_data = []

        target_real_label = []
        class_index_target = []

        domain_label = []
        ST_split = [] # Mask of targets to be evaluated
        # select index for support class
        num_class_index_target = int(self.target_ratio * (self.num_class - 1))

        if self.target_ratio > 0:
            available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0
                               and key < self.num_class - 1]
            class_index_target = random.sample(available_index, min(num_class_index_target, len(available_index)))

        class_index_source = list(set(range(self.num_class - 1)) - set(class_index_target))
        random.shuffle(class_index_source)

        for classes in class_index_source:
            # select support samples from source domain or target domain
            image = Image.open(random.choice(self.source_image[classes])).convert('RGB')

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            domain_label.append(1)
            ST_split.append(0)
            # target_real_label.append(classes)
        for classes in class_index_target:
            # select support samples from source domain or target domain
            image = Image.open(random.choice(self.target_image_list[classes])).convert('RGB')

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            domain_label.append(0)
            ST_split.append(0)
            # target_real_label.append(classes)

        # adding target samples
        for i in range(self.num_class - 1):

            if self.partition == 'train':
                if self.target_ratio > 0:
                    index = random.choice(list(range(len(self.label_flag))))
                else:
                    index = random.choice(list(range(len(self.target_image))))
                # index = random.choice(list(range(len(self.label_flag))))
                target_image = Image.open(self.target_image[index]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(self.label_flag[index])
                target_real_label.append(self.target_label[index])
                domain_label.append(0)
                ST_split.append(1)
            elif self.partition == 'test':
                # For last batch
                # if item * (self.num_class - 1) + i >= len(self.target_image):
                #     break
                target_image = Image.open(self.target_image[item * (self.num_class - 1) + i]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(self.num_class)
                target_real_label.append(self.target_label[item * (self.num_class - 1) + i])
                domain_label.append(0)
                ST_split.append(1)
        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)
        real_label_data = torch.tensor(target_real_label)
        domain_label = torch.tensor(domain_label)
        ST_split = torch.tensor(ST_split)
        return image_data, label_data, real_label_data, domain_label, ST_split

    def load_dataset(self):
        source_image_list = {key: [] for key in range(self.num_class - 1)}
        target_image_list = []
        target_label_list = []
        with open(self.source_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                if label == str(self.num_class-1):
                    continue
                source_image_list[int(label)].append(image_dir)
                # source_image_list.append(image_dir)

        with open(self.target_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                # target_image_list[int(label)].append(image_dir)
                target_image_list.append(image_dir)
                target_label_list.append(int(label))

        return source_image_list, target_image_list, target_label_list

class Digits_Dataset(Multi_Source_Base_Dataset):

    def __init__(self, args, partition):
        super(Digits_Dataset, self).__init__(args, partition)
        # set dataset info

        self.class_name = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.domain_name = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
        self.target = args.target_domain
        self.unk_class = args.unk_class
        self.openset = True if args.unk_class is not None else False
        self.num_class = args.unk_class + 1 if self.openset else len(self.class_name)
        self.num_domains = len(self.domain_name)
        self.mean_pix = torch.tensor((0.,))
        self.std_pix = torch.tensor((1.,))
        self.scale = 32
        self.crop_scale = 32
        self.partition = partition
        self.args = args
        normalize = transforms.Normalize(self.mean_pix, self.std_pix)
        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   # transforms.RandomHorizontalFlip(),
                                                   # transforms.RandomRotation(10),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   # transforms.CenterCrop(self.crop_scale),
                                                   transforms.ToTensor(),
                                                   normalize])

        self.source_data, self.target_data = self.load_multi_datasets(self.target)

    def __len__(self):
        full_set = self.source_data + self.target_data
        return min([len(full_set[domain]["labels"]) for domain in range(len(full_set))])

    def __getitem__(self, item):
        image_data = []
        label_data = []
        domain_labels = []
        for source_id in range(self.num_domains - 1):
            image = self.source_data[source_id]["imgs"][item].transpose(1, 2, 0)
            label = self.source_data[source_id]["labels"][item]

            if list(image.shape)[-1] == 3:
                image = Image.fromarray(np.uint8(image)).convert('RGB')
            elif list(image.shape)[-1] == 1:
                # test = Image.fromarray(np.uint8(image.squeeze()))
                image = Image.fromarray(np.uint8(np.repeat(image, 3, axis=-1)))
            else:
                print("Domain {} Image #{} error".format(source_id, item))
            # image.show()
            if self.transformer is not None:
                image = self.transformer(image)

            image_data.append(image)
            label_data.append(label)
            domain_labels.append(source_id)

        image = self.target_data[0]["imgs"][item].transpose(1, 2, 0)
        label = self.target_data[0]["labels"][item]
        if list(image.shape)[-1] == 3:
            image = Image.fromarray(np.uint8(image)).convert('RGB')
        elif list(image.shape)[-1] == 1:
            # test = Image.fromarray(np.uint8(image.squeeze()))
            image = Image.fromarray(np.uint8(np.repeat(image, 3, axis=-1)))
        if self.transformer is not None:
            image = self.transformer(image)
        image_data.append(image)
        label_data.append(label)
        domain_labels.append(self.num_domains - 1)

        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)
        domain_labels = torch.tensor(domain_labels)

        return image_data, label_data, domain_labels


    def shuffle(self, data):
        # new_data = {}
        zip_data = list(zip(data["imgs"], data["labels"]))
        random.shuffle(zip_data)
        shuffled_imgs, shuffled_labels = zip(*zip_data)
        new_data = {"imgs": shuffled_imgs, "labels": shuffled_labels}
        return new_data

    def shuffle_datasets(self):
        self.target_data[0] = self.shuffle(self.target_data[0])
        for source_id in range(self.num_domains - 1):
            self.source_data[source_id] = self.shuffle(self.source_data[source_id])

    def load_dataset(self, domain_name):

        if domain_name == 'svhn':
            train_image, train_label, \
            test_image, test_label = load_svhn(self.args)
        if domain_name == 'mnist':
            train_image, train_label, \
            test_image, test_label = load_mnist(self.args)

        if domain_name == 'mnistm':
            train_image, train_label, \
            test_image, test_label = load_mnistm(self.args)

        if domain_name == 'usps':
            train_image, train_label, \
            test_image, test_label = load_usps(self.args)

        if domain_name == 'syn':
            train_image, train_label, \
            test_image, test_label = load_syn(self.args)

        return train_image, train_label, test_image, test_label

    def load_multi_datasets(self, target):
        source_data_list = []
        target_data_list = []

        source_name = self.domain_name
        source_name.remove(target)
        if self.partition == "train":
            target_train, target_train_label, _, _ = self.load_dataset(target)
            if self.openset:
                target_train_label[target_train_label >= self.unk_class] = self.unk_class

            target_data_list.append({"imgs": target_train, "labels": target_train_label})
            for source_id, source in enumerate(source_name):
                source_train, source_train_label, _, _ = self.load_dataset(source)
                if self.openset:
                    train_idx_del = np.array([source_train_label[j] == source_id or source_train_label[j] >= self.unk_class
                                              for j in range(len(source_train_label))])
                    source_train = source_train[~train_idx_del]
                    source_train_label = source_train_label[~train_idx_del]
                source_data_list.append({"imgs": source_train, "labels": source_train_label})

        elif self.partition == "test":
            _, _, target_test, target_test_label = self.load_dataset(target)
            if self.openset:
                target_test_label[target_test_label >= self.unk_class] = self.unk_class

            target_length = len(target_test_label)
            target_data_list.append({"imgs": target_test, "labels": target_test_label})
            for source_id, source in enumerate(source_name):
                # _, _, source_test, source_test_label = self.load_dataset(source)

                # get more source samples to match the target samples
                # if len(source_test_label) < target_length:
                #     idx = random.choices(range(len(source_test_label)), k=target_length-len(source_test_label))
                #     source_test = np.concatenate((source_test, source_test[idx]), axis=0)
                #     source_test_label = np.concatenate((source_test_label, source_test_label[idx]))
                # if self.openset:
                #     test_idx_del = np.array([source_test_label[j] == source_id or source_test_label[j] >= self.unk_class
                #                              for j in range(len(source_test_label))])
                #     source_test = source_test[~test_idx_del]
                #     source_test_label = source_test_label[~test_idx_del]
                source_data_list.append({"imgs": target_test, "labels": target_test_label})

        return source_data_list, target_data_list

class Office_Dataset(Multi_Source_Base_Dataset):

    def __init__(self, args, partition):
        super(Office_Dataset, self).__init__(args, partition)
        # set dataset info
        self.args = args
        self.class_name = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle",
                           "calculator", "desk_chair", "desk_lamp", "desktop_computer", "file_cabinet", "unk"]
        self.domain_name = ['amazon', 'dslr', 'webcam']
        self.target = args.target_domain
        self.unk_class = args.unk_class
        self.openset = True if args.unk_class is not None else False
        self.rm_idx = [[0, 1, 2, 3], [4, 5, 6, 7]] if args.strict_setting else None
        self.num_class = len(self.class_name)
        self.num_domains = len(self.domain_name)
        self.mean_pix = torch.tensor((0.,))
        self.std_pix = torch.tensor((1.,))
        self.scale = 224
        self.partition = partition

        normalize = transforms.Normalize(self.mean_pix, self.std_pix)
        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   # transforms.RandomHorizontalFlip(),
                                                   # transforms.RandomRotation(10),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   # transforms.CenterCrop(self.crop_scale),
                                                   transforms.ToTensor(),
                                                   normalize])

        self.source_data, self.target_data = self.load_multi_datasets(self.target)

    def __len__(self):
        full_set = self.source_data + self.target_data
        return min([len(full_set[domain]["labels"]) for domain in range(len(full_set))])

    def __getitem__(self, item):
        image_data = []
        label_data = []
        domain_labels = []
        for source_id in range(self.num_domains - 1):
            image = Image.open(self.source_data[source_id]["imgs"][item]).convert('RGB')
            label = self.source_data[source_id]["labels"][item]

            if self.transformer is not None:
                image = self.transformer(image)

            image_data.append(image)
            label_data.append(label)
            domain_labels.append(source_id)

        image = Image.open(self.target_data[0]["imgs"][item]).convert('RGB')
        label = self.target_data[0]["labels"][item]

        if self.transformer is not None:
            image = self.transformer(image)
        image_data.append(image)
        label_data.append(label)
        domain_labels.append(self.num_domains - 1)

        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)
        domain_labels = torch.tensor(domain_labels)

        return image_data, label_data, domain_labels

    def shuffle(self, data):
        # new_data = {}
        zip_data = list(zip(data["imgs"], data["labels"]))
        random.shuffle(zip_data)
        shuffled_imgs, shuffled_labels = zip(*zip_data)
        new_data = {"imgs": shuffled_imgs, "labels": shuffled_labels}
        return new_data

    def shuffle_datasets(self):
        self.target_data[0] = self.shuffle(self.target_data[0])
        for source_id in range(self.num_domains - 1):
            self.source_data[source_id] = self.shuffle(self.source_data[source_id])

    def load_dataset(self, domain_name):
        args = self.args

        train_image, train_label, test_image, test_label = load_office(args, domain_name)

        return train_image, train_label, test_image, test_label

    def load_multi_datasets(self, target):
        source_data_list = []
        target_data_list = []

        source_name = self.domain_name
        source_name.remove(target)
        if self.partition == "train":
            target_train, target_train_label, _, _ = self.load_dataset(target)
            target_data_list.append({"imgs": target_train, "labels": target_train_label})
            for source_id, source in enumerate(source_name):
                source_train, source_train_label, _, _ = self.load_dataset(source)
                if self.openset:
                    if self.args.strict_setting:
                        train_idx_del = np.array(
                            [source_train_label[j] in self.rm_idx[source_id] for j in range(len(source_train_label))])
                    else:
                        train_idx_del = np.array([source_train_label[j] == source_id for j in range(len(source_train_label))])
                    idx = np.where(train_idx_del == False)
                    source_train = [source_train[i] for i in idx[0]]
                    source_train_label = source_train_label[idx] # @note 0,1,2,3 as unk
                source_data_list.append({"imgs": source_train, "labels": source_train_label})

        elif self.partition == "test":
            _, _, target_test, target_test_label = self.load_dataset(target)

            target_data_list.append({"imgs": target_test, "labels": target_test_label})
            for source_id, source in enumerate(source_name):
                source_data_list.append({"imgs": target_test, "labels": target_test_label})

        return source_data_list, target_data_list

class DomainNet_Dataset(Multi_Source_Base_Dataset):

    def __init__(self, args, partition):
        super(DomainNet_Dataset, self).__init__(args, partition)
        # set dataset info
        self.args = args
        self.class_name = list(np.arange(345))
        self.domain_name = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        self.target = args.target
        self.unk_class = args.unk_class
        self.openset = True if args.unk_class is not None else False
        self.rm_idx = [list(np.arange(50)), list(np.arange(50, 100)), list(np.arange(100, 150)), list(np.arange(150, 200)),
                       list(np.arange(200, 250))] if args.strict_setting else None
        self.num_class = self.unk_class + 1
        self.num_domains = len(self.domain_name)
        self.mean_pix = torch.tensor((0.,))
        self.std_pix = torch.tensor((1.,))
        self.scale = 32
        self.partition = partition

        normalize = transforms.Normalize(self.mean_pix, self.std_pix)
        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(size=(self.scale, self.scale)),
                                                   # transforms.RandomHorizontalFlip(),
                                                   # transforms.RandomRotation(10),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   # transforms.CenterCrop(self.crop_scale),
                                                   transforms.ToTensor(),
                                                   normalize])

        self.source_data, self.target_data = self.load_multi_datasets(self.target)

    def __len__(self):
        full_set = self.source_data + self.target_data
        return min([len(full_set[domain]["labels"]) for domain in range(len(full_set))])

    def __getitem__(self, item):
        image_data = []
        label_data = []
        domain_labels = []
        for source_id in range(self.num_domains - 1):
            image = Image.open(self.source_data[source_id]["imgs"][item]).convert('RGB')
            label = self.source_data[source_id]["labels"][item]

            if self.transformer is not None:
                image = self.transformer(image)

            image_data.append(image)
            label_data.append(label)
            domain_labels.append(source_id)

        image = Image.open(self.target_data[0]["imgs"][item]).convert('RGB')
        label = self.target_data[0]["labels"][item]

        if self.transformer is not None:
            image = self.transformer(image)
        image_data.append(image)
        label_data.append(label)
        domain_labels.append(self.num_domains - 1)

        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)
        domain_labels = torch.tensor(domain_labels)

        return image_data, label_data, domain_labels

    def shuffle(self, data):
        # new_data = {}
        zip_data = list(zip(data["imgs"], data["labels"]))
        random.shuffle(zip_data)
        shuffled_imgs, shuffled_labels = zip(*zip_data)
        new_data = {"imgs": shuffled_imgs, "labels": shuffled_labels}
        return new_data

    def shuffle_datasets(self):
        self.target_data[0] = self.shuffle(self.target_data[0])
        for source_id in range(self.num_domains - 1):
            self.source_data[source_id] = self.shuffle(self.source_data[source_id])

    def load_dataset(self, domain_name):
        args = self.args

        train_image, train_label, test_image, test_label = load_domainnet(args, domain_name)

        return train_image, train_label, test_image, test_label

    def load_multi_datasets(self, target):
        source_data_list = []
        target_data_list = []

        source_name = self.domain_name
        source_name.remove(target)
        if self.partition == "train":
            target_train, target_train_label, _, _ = self.load_dataset(target)
            if self.openset:
                target_train_label[target_train_label >= self.unk_class] = self.unk_class
            target_data_list.append({"imgs": target_train, "labels": target_train_label})
            for source_id, source in enumerate(source_name):
                source_train, source_train_label, _, _ = self.load_dataset(source)
                if self.openset:
                    if self.args.strict_setting:
                        train_idx_del = np.array(
                            [(source_train_label[j] in self.rm_idx[source_id] or source_train_label[j] >= self.unk_class) for j in range(len(source_train_label))])
                    else:
                        train_idx_del = np.array([source_train_label[j] == source_id for j in range(len(source_train_label))])
                    idx = np.where(train_idx_del == False)
                    source_train = [source_train[i] for i in idx[0]]
                    source_train_label = source_train_label[idx]
                source_data_list.append({"imgs": source_train, "labels": source_train_label})

        elif self.partition == "test":
            _, _, target_test, target_test_label = self.load_dataset(target)
            if self.openset:
                target_test_label[target_test_label >= self.unk_class] = self.unk_class
            target_data_list.append({"imgs": target_test, "labels": target_test_label})
            for source_id, source in enumerate(source_name):
                source_data_list.append({"imgs": target_test, "labels": target_test_label})

        return source_data_list, target_data_list


class Base_Dataset(data.Dataset):
    def __init__(self, root, partition, target_ratio=0.0):
        super(Base_Dataset, self).__init__()
        # set dataset info
        self.root = root
        self.partition = partition
        self.target_ratio = target_ratio
        # self.target_ratio=0 no mixup
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(224),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   normalize])

    def __len__(self):

        if self.partition == 'train':
            return int(min(sum(self.alpha), len(self.target_image)) / (self.num_class - 1))
        elif self.partition == 'test':
            return int(len(self.target_image) / (self.num_class - 1))

    def __getitem__(self, item):

        image_data = []
        label_data = []

        target_real_label = []
        class_index_target = []

        domain_label = []
        ST_split = [] # Mask of targets to be evaluated
        # select index for support class
        num_class_index_target = int(self.target_ratio * (self.num_class - 1))

        if self.target_ratio > 0:
            available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0
                               and key < self.num_class - 1]
            class_index_target = random.sample(available_index, min(num_class_index_target, len(available_index)))

        class_index_source = list(set(range(self.num_class - 1)) - set(class_index_target))
        random.shuffle(class_index_source)

        for classes in class_index_source:
            # select support samples from source domain or target domain
            image = Image.open(random.choice(self.source_image[classes])).convert('RGB')

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            domain_label.append(1)
            ST_split.append(0)
            # target_real_label.append(classes)
        for classes in class_index_target:
            # select support samples from source domain or target domain
            image = Image.open(random.choice(self.target_image_list[classes])).convert('RGB')

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            domain_label.append(0)
            ST_split.append(0)
            # target_real_label.append(classes)

        # adding target samples
        for i in range(self.num_class - 1):

            if self.partition == 'train':
                if self.target_ratio > 0:
                    index = random.choice(list(range(len(self.label_flag))))
                else:
                    index = random.choice(list(range(len(self.target_image))))
                # index = random.choice(list(range(len(self.label_flag))))
                target_image = Image.open(self.target_image[index]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(self.label_flag[index])
                target_real_label.append(self.target_label[index])
                domain_label.append(0)
                ST_split.append(1)
            elif self.partition == 'test':
                # For last batch
                # if item * (self.num_class - 1) + i >= len(self.target_image):
                #     break
                target_image = Image.open(self.target_image[item * (self.num_class - 1) + i]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(self.num_class)
                target_real_label.append(self.target_label[item * (self.num_class - 1) + i])
                domain_label.append(0)
                ST_split.append(1)
        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)
        real_label_data = torch.tensor(target_real_label)
        domain_label = torch.tensor(domain_label)
        ST_split = torch.tensor(ST_split)
        return image_data, label_data, real_label_data, domain_label, ST_split

    def load_dataset(self):
        source_image_list = {key: [] for key in range(self.num_class - 1)}
        target_image_list = []
        target_label_list = []
        with open(self.source_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                if label == str(self.num_class-1):
                    continue
                source_image_list[int(label)].append(image_dir)
                # source_image_list.append(image_dir)

        with open(self.target_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                # target_image_list[int(label)].append(image_dir)
                target_image_list.append(image_dir)
                target_label_list.append(int(label))

        return source_image_list, target_image_list, target_label_list


class PACS_Dataset(Multi_Source_Base_Dataset):
    
    def __init__(self, args, partition):
        super(PACS_Dataset, self).__init__(args, partition)
        # set dataset info
        self.args = args
        # @todo  
        self.class_name = ["0", "1", "2", "3", "4", "5"]   # need to remove some of them as unk  
            
        self.domain_name = ['art_painting', 'cartoon', 'photo', 'sketch']
        self.target = args.target_domain
        self.unk_class = args.unk_class
        self.openset = True if args.unk_class is not None else False
        # @note  this is for remove some of the classes in domains to make it as openset setting, the method could be vary
        self.rm_idx = [[0], [1], [2]] if args.strict_setting else None
        self.num_class = len(self.class_name)
        self.num_domains = len(self.domain_name)
        self.mean_pix = torch.tensor((0.,))
        self.std_pix = torch.tensor((1.,))
        self.scale = 64
        self.partition = partition
        # self.crop_scale = 224 # zixin: define new values for crop_scale

        normalize = transforms.Normalize(self.mean_pix, self.std_pix)
        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(self.scale),
                                                   transforms.ToTensor(),
                                                   normalize])

        self.source_data, self.target_data = self.load_multi_datasets(self.target)

    def __len__(self):
        full_set = self.source_data + self.target_data
        return min([len(full_set[domain]["labels"]) for domain in range(len(full_set))])

    def __getitem__(self, item):
        image_data = []
        label_data = []
        domain_labels = []
        for source_id in range(self.num_domains - 1):
            image = Image.open(self.source_data[source_id]["imgs"][item]).convert('RGB')
            label = self.source_data[source_id]["labels"][item]

            if self.transformer is not None:
                image = self.transformer(image)

            image_data.append(image)
            label_data.append(label)
            domain_labels.append(source_id)

        image = Image.open(self.target_data[0]["imgs"][item]).convert('RGB')
        label = self.target_data[0]["labels"][item]

        if self.transformer is not None:
            image = self.transformer(image)
        image_data.append(image)
        label_data.append(label)
        domain_labels.append(self.num_domains - 1)

        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)
        domain_labels = torch.tensor(domain_labels)

        return image_data, label_data, domain_labels

    def shuffle(self, data):
        # new_data = {}
        zip_data = list(zip(data["imgs"], data["labels"]))
        random.shuffle(zip_data)
        shuffled_imgs, shuffled_labels = zip(*zip_data)
        new_data = {"imgs": shuffled_imgs, "labels": shuffled_labels}
        return new_data

    def shuffle_datasets(self):
        self.target_data[0] = self.shuffle(self.target_data[0])
        for source_id in range(self.num_domains - 1):
            self.source_data[source_id] = self.shuffle(self.source_data[source_id])

    def load_dataset(self, domain_name):
        args = self.args
        train_image, train_label, test_image, test_label = load_pacs(args, domain_name)
        return train_image, train_label, test_image, test_label

    def load_multi_datasets(self, target):
        source_data_list = []
        target_data_list = []

        source_name = self.domain_name  # 4 domains
        source_name.remove(target)  # under multi-source setting, we still have 3 domains
        if self.partition == "train":
            target_train, target_train_label, _, _ = self.load_dataset(target)
            # target_train_label = np.array([x-1 for x in target_train_label])

            target_data_list.append({"imgs": target_train, "labels": target_train_label})
            for source_id, source in enumerate(source_name):
                source_train, source_train_label, _, _ = self.load_dataset(source)
                if self.openset:
                    # source and target only share a part of data classes
                    if self.args.strict_setting:
                        train_idx_del = np.array(
                            [source_train_label[j] in self.rm_idx[source_id] for j in range(len(source_train_label))])
                    else:
                        train_idx_del = np.array([source_train_label[j] == source_id for j in range(len(source_train_label))])
                    idx = np.where(train_idx_del == False)
                    source_train = [source_train[i] for i in idx[0]]
                    source_train_label = source_train_label[idx]
                    # source_train_label = np.array([x-1 for x in source_train_label])
                source_data_list.append({"imgs": source_train, "labels": source_train_label})

        elif self.partition == "test":
            _, _, target_test, target_test_label = self.load_dataset(target)
            # target_test_label = np.array([x-1 for x in target_test_label])
            target_data_list.append({"imgs": target_test, "labels": target_test_label})
            for source_id, source in enumerate(source_name):
                source_data_list.append({"imgs": target_test, "labels": target_test_label})

        return source_data_list, target_data_list


class CIFAR_Dataset(Multi_Source_Base_Dataset):
        
    def __init__(self, args, partition):
        super(CIFAR_Dataset, self).__init__(args, partition)
        # set dataset info
        self.args = args
        self.scale = 32

        self.class_name = ['0','1','2','3','4','5']
        self.domain_name = ['brightness', 'contrast', 'fog', 'defocus_blur', 'frost'] 
        # , 'gaussian_blur', 'gaussian_noise', 'elastic_transform',
        #  'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 
        #  'spatter', 'speckle_noise', 'zoom_blur'

        self.target = args.target_domain
        self.unk_class = args.unk_class
        self.openset = True if args.unk_class is not None else False

        # @note  this is for remove some of the classes in domains to make it as openset setting, the method could be vary
        self.rm_idx = [[0], [1], [2], [3]] if args.strict_setting else None
        self.num_class = len(self.class_name)
        self.num_domains = len(self.domain_name)

        self.mean_pix = torch.tensor((0.,))
        self.std_pix = torch.tensor((1.,))
        self.scale = 32
        self.partition = partition
        # self.crop_scale = 224 # zixin: define new values for crop_scale

        normalize = transforms.Normalize(self.mean_pix, self.std_pix)
        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(self.scale),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(self.scale),
                                                   normalize])

        self.source_data, self.target_data = self.load_multi_datasets(self.target)

    def __len__(self):
        full_set = self.source_data + self.target_data
        return min([len(full_set[domain]["labels"]) for domain in range(len(full_set))])

    def __getitem__(self, item):
        image_data = []
        label_data = []
        domain_labels = []
        for source_id in range(self.num_domains - 1):
            image = self.source_data[source_id]["imgs"][item]
            label = self.source_data[source_id]["labels"][item]

            if self.transformer is not None:
                image = self.transformer(image)

            image_data.append(image)
            label_data.append(label)
            domain_labels.append(source_id)

        image = self.target_data[0]["imgs"][item]
        label = self.target_data[0]["labels"][item]

        if self.transformer is not None:
            image = self.transformer(image)
        image_data.append(image)
        label_data.append(label)
        domain_labels.append(self.num_domains - 1)

        image_data = torch.stack(image_data)
        label_data = torch.LongTensor(label_data)
        domain_labels = torch.tensor(domain_labels)

        return image_data, label_data, domain_labels

    def shuffle(self, data):
        # new_data = {}
        zip_data = list(zip(data["imgs"], data["labels"]))
        random.shuffle(zip_data)
        shuffled_imgs, shuffled_labels = zip(*zip_data)
        new_data = {"imgs": shuffled_imgs, "labels": shuffled_labels}
        return new_data

    def shuffle_datasets(self):
        self.target_data[0] = self.shuffle(self.target_data[0])
        for source_id in range(self.num_domains - 1):
            self.source_data[source_id] = self.shuffle(self.source_data[source_id])

    def load_dataset(self, domain_name):
        args = self.args
        train_image, train_label, test_image, test_label = load_cifar(args, domain_name)
        return train_image, train_label, test_image, test_label

    def load_multi_datasets(self, target):
        source_data_list = []
        target_data_list = []

        source_name = self.domain_name  
        source_name.remove(target)  
        if self.partition == "train":
            target_train, target_train_label, _, _ = self.load_dataset(target)
            target_data_list.append({"imgs": target_train, "labels": target_train_label})
            for source_id, source in enumerate(source_name):
                source_train, source_train_label, _, _ = self.load_dataset(source)
                if self.openset:
                    # source and target only share a part of data classes
                    if self.args.strict_setting:
                        train_idx_del = np.array(
                            [source_train_label[j] in self.rm_idx[source_id] for j in range(len(source_train_label))])
                    else:
                        train_idx_del = np.array([source_train_label[j] == source_id for j in range(len(source_train_label))])
                    idx = np.where(train_idx_del == False)
                    source_train = [source_train[i] for i in idx[0]]
                    source_train_label = source_train_label[idx]
                source_data_list.append({"imgs": source_train, "labels": source_train_label})

        elif self.partition == "test":
            _, _, target_test, target_test_label = self.load_dataset(target)

            target_data_list.append({"imgs": target_test, "labels": target_test_label})
            for source_id, source in enumerate(source_name):
                source_data_list.append({"imgs": target_test, "labels": target_test_label})

        return source_data_list, target_data_list


# class Office_Dataset(Base_Dataset):
#
#     def __init__(self, root, partition, label_flag=None, source='A', target='W', target_ratio=0.0):
#         super(Office_Dataset, self).__init__(root, partition, target_ratio)
#         # set dataset info
#         src_name, tar_name = self.getFilePath(source, target)
#         self.source_path = os.path.join(root, src_name)
#         self.target_path = os.path.join(root, tar_name)
#         self.class_name = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle",
#                            "calculator", "desk_chair", "desk_lamp", "desktop_computer", "file_cabinet", "unk"]
#         self.num_class = len(self.class_name)
#         self.source_image, self.target_image, self.target_label = self.load_dataset()
#         self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
#         self.label_flag = label_flag
#
#         # create the unlabeled tag
#         if self.label_flag is None:
#             self.label_flag = torch.ones(len(self.target_image)) * self.num_class
#
#         else:
#             # if pseudo label comes
#             self.target_image_list = {key: [] for key in range(self.num_class + 1)}
#             for i in range(len(self.label_flag)):
#                 self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
#
#         if self.target_ratio > 0:
#             self.alpha_value = [len(self.source_image[key]) + len(self.target_image_list[key]) for key in self.source_image.keys()]
#         else:
#             self.alpha_value = self.alpha
#
#         self.alpha_value = np.array(self.alpha_value)
#         self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
#         self.alpha_value = torch.tensor(self.alpha_value).float().cuda()
#
#     def getFilePath(self, source, target):
#
#         if source == 'A':
#             src_name = 'amazon_src_list.txt'
#         elif source == 'W':
#             src_name = 'webcam_src_list.txt'
#         elif source == 'D':
#             src_name = 'dslr_src_list.txt'
#         else:
#             print("Unknown Source Type, only supports A W D.")
#
#         if target == 'A':
#             tar_name = 'amazon_tar_list.txt'
#         elif target == 'W':
#             tar_name = 'webcam_tar_list.txt'
#         elif target == 'D':
#             tar_name = 'dslr_tar_list.txt'
#         else:
#             print("Unknown Target Type, only supports A W D.")
#
#         return src_name, tar_name
#
#
#
#
# class Home_Dataset(Base_Dataset):
#     def __init__(self, root, partition, label_flag=None, source='A', target='R', target_ratio=0.0):
#         super(Home_Dataset, self).__init__(root, partition, target_ratio)
#         src_name, tar_name = self.getFilePath(source, target)
#         self.source_path = os.path.join(root, src_name)
#         self.target_path = os.path.join(root, tar_name)
#         self.class_name = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
#                            'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
#                            'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
#                            'Fork', 'unk']
#         self.num_class = len(self.class_name)
#
#         self.source_image, self.target_image, self.target_label = self.load_dataset()
#         self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
#         self.label_flag = label_flag
#
#         # create the unlabeled tag
#         if self.label_flag is None:
#             self.label_flag = torch.ones(len(self.target_image)) * self.num_class
#
#         else:
#             # if pseudo label comes
#             self.target_image_list = {key: [] for key in range(self.num_class + 1)}
#             for i in range(len(self.label_flag)):
#                 self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
#
#         # if self.target_ratio > 0:
#         #     self.alpha_value = [len(self.source_image[key]) + len(self.target_image_list[key]) for key in
#         #                         self.source_image.keys()]
#         # else:
#         #     self.alpha_value = self.alpha
#         #
#         # self.alpha_value = np.array(self.alpha_value)
#         # self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
#         # self.alpha_value = torch.tensor(self.alpha_value).float().cuda()
#
#     def getFilePath(self, source, target):
#
#         if source == 'A':
#             src_name = 'art_source.txt'
#         elif source == 'C':
#             src_name = 'clip_source.txt'
#         elif source == 'P':
#             src_name = 'product_source.txt'
#         elif source == 'R':
#             src_name = 'real_source.txt'
#         else:
#             print("Unknown Source Type, only supports A C P R.")
#
#         if target == 'A':
#             tar_name = 'art_tar.txt'
#         elif target == 'C':
#             tar_name = 'clip_tar.txt'
#         elif target == 'P':
#             tar_name = 'product_tar.txt'
#         elif target == 'R':
#             tar_name = 'real_tar.txt'
#         else:
#             print("Unknown Target Type, only supports A C P R.")
#
#         return src_name, tar_name
#
#
# class Visda_Dataset(Base_Dataset):
#     def __init__(self, root, partition, label_flag=None, target_ratio=0.0):
#         super(Visda_Dataset, self).__init__(root, partition, target_ratio)
#         # set dataset info
#         self.source_path = os.path.join(root, 'source_list.txt')
#         self.target_path = os.path.join(root, 'target_list.txt')
#         self.class_name = ["bicycle", "bus", "car", "motorcycle", "train", "truck", 'unk']
#         self.num_class = len(self.class_name)
#         self.source_image, self.target_image, self.target_label = self.load_dataset()
#         self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
#         self.label_flag = label_flag
#
#         # create the unlabeled tag
#         if self.label_flag is None:
#             self.label_flag = torch.ones(len(self.target_image)) * self.num_class
#
#         else:
#             # if pseudo label comes
#             self.target_image_list = {key: [] for key in range(self.num_class + 1)}
#             for i in range(len(self.label_flag)):
#                 self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
#
# class Visda18_Dataset(Base_Dataset):
#     def __init__(self, root, partition, label_flag=None, target_ratio=0.0):
#         super(Visda18_Dataset, self).__init__(root, partition, target_ratio)
#         # set dataset info
#         self.source_path = os.path.join(root, 'source_list_k.txt')
#         self.target_path = os.path.join(root, 'target_list.txt')
#         self.class_name = ["areoplane","bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant",
#                            "skateboard", "train", "truck", 'unk']
#         self.num_class = len(self.class_name)
#         self.source_image, self.target_image, self.target_label = self.load_dataset()
#         self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
#         self.label_flag = label_flag
#
#         # create the unlabeled tag
#         if self.label_flag is None:
#             self.label_flag = torch.ones(len(self.target_image)) * self.num_class
#
#         else:
#             # if pseudo label comes
#             self.target_image_list = {key: [] for key in range(self.num_class + 1)}
#             for i in range(len(self.label_flag)):
#                 self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])



# train_idx_del = np.array([source_train_label[j] in rm_idx[source_id] for j in range(len(source_train_label))])