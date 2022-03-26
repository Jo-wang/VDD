import os
import numpy as np    
import torch
import random

# from torch._C import float32

domain = ['brightness', 'contrast', 'fog', 'defocus_blur', 'frost', 'gaussian_blur', 'gaussian_noise', 'elastic_transform',
         'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 
         'spatter', 'speckle_noise', 'zoom_blur']

labels = 'labels.npy'

data_type = '.npy'

abs_path = '/media/bigdata/zixin/CIFAR-10-C/'

shared = ['0', '1', '2', '3', '4']
unk = ['5', '6', '7', '8', '9']

levels = [1, 2, 3, 4, 5]

# @note only choose level=1 images and labels (the smallest noise) the range is (0, 10000)
def load_cifar(args, data):
    file_name = abs_path + data + data_type
    dataset = np.load(file_name, allow_pickle=True, encoding="latin1").tolist()
    labels = np.load(abs_path + 'labels.npy', allow_pickle=True, encoding="latin1").tolist()
    train_idx = []
    train_data = []
    train_label = []
    for i in range(len(labels)):
        if i < 10000:
            if str(labels[i]) not in unk:
                train_idx.append(i)
                train_label.append(labels[i])
    for data_idx in range(len(dataset)):
        if data_idx in train_idx:
            train_data.append(dataset[data_idx])
    
    test_data = []
    for i in range(len(dataset)):
        if i < 10000:
            test_data.append(dataset[i])
    test_label = []
    for j in range(len(labels)):
        # @note since we have 5 levels in each domain, we only choose one level of them, so this should be: total_num(50000)/5 = 10000
        if j < 10000:   
            if str(labels[j]) in unk:
                labels[j] = 5
                # train_idx.append(i)
            test_label.append(labels[j])
    return np.array(train_data, dtype='float32'), np.array(train_label), np.array(test_data, dtype='float32'), np.array(test_label)

# a = load_cifar(False, "fog")


# data = np.load(abs_path+'fog'+data_type).tolist()

# labels = np.load(abs_path+'labels.npy', encoding="bytes").tolist()


# print(set(labels))
# #     if idx == 0:
# #         break
# import matplotlib.pyplot as plt
# data = np.load(abs_path + 'brightness' + data_type, encoding='bytes').tolist()
# plt.imshow(data[40001])
# plt.savefig("img40001a.jpg")
# # plt.imshow(data[10000])

# # label = []

# def load_cifar1(source, target):
#     # @note load source data and label (remove all unk)
#     source_data = []
#     source_labels = []
#     labels = np.load(abs_path + 'labels.npy', encoding="bytes").tolist()
#     for idx, domain in enumerate(source):
#         file = abs_path + domain + data_type
#         shared_idx = []
        
#         for id, label in enumerate(labels):
#             label = str(label)
#             if label in shared:
#                 shared_idx.append(id)
#                 source_labels.append(label)
#         # source_domain_label = domain
#         data = np.load(file, encoding="bytes").tolist()
#         # shared_data = []
#         for id1, d in enumerate(data):
#             if id1 in shared_idx:
#                 source_data.append(d)
#         # num_data = len(data)

#     # @note load target data and labels (contains unk)
#     target_data = np.load(abs_path + target[0] + data_type, encoding="bytes").tolist()
#     target_labels = []
#     for lab in labels:
#         lab = str(lab)
#         if lab in unk:
#             lab = '8'
#         target_labels.append(lab)
    
#     # @note change source/target data to train/test data
#     train_data_target = []
#     train_label_target = []
#     for i, j in zip(target_data, target_labels):
#         if j != '8':
#             train_data_target.append(i)
#             train_label_target.append(j)
#     train_data = source_data + train_data_target
#     train_label = source_labels + train_label_target
#     test_data = source_data + target_data
#     test_label = source_labels + target_labels
#     return train_data, train_label, test_data, test_label