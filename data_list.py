# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import math


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        # if len(imgs) == 0:
        #     raise(RuntimeError("Found 0 image in subfolders of: " + root + "\n"
        #                        "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs*50
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        # if len(imgs) == 0:
        #     raise(RuntimeError("Found 0 image in subfolders of: " + root + "\n"
        #                        "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class DsetThreeChannels(Dataset):
    # Make sure that your dataset actually returns images with only one channel!

    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        image, label = self.dset[index]
        return image.repeat(3, 1, 1), label

    def __len__(self):
        return len(self.dset)

class data_batch:
    def __init__(self, gt_data, batch_size: int, drop_last: bool, gt_flag: bool, num_class: int, num_batch: int) -> None:
        # if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
        #         batch_size <= 0:
        #     raise ValueError("batch_size should be a positive integer value, "
        #                      "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        gt_data = gt_data.astype(dtype=int)

        self.class_num = num_class
        self.batch_num = num_batch

        self.random_loader = False
        self.all_data = np.arange(len(gt_data))
        self.data_len = len(gt_data)

        self.rl_len= math.floor(self.data_len/ self.batch_num)

        self.i_range = len(gt_data)
        self.s_list = []
        if gt_flag == False:
            self.random_loader = True
            self.set_length(self.rl_len)
            self.i_range = self.rl_len
        else:
            for c_iter in range(self.class_num):
                cur_data = np.where(gt_data == c_iter)[0]
                self.s_list.append(cur_data)
                cur_length = math.floor((len(cur_data) * self.class_num) / self.batch_num)
                if(cur_length < self.data_len):
                    self.set_length(cur_length)
                    self.i_range = len(cur_data)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.prob_mat = np.zeros(())
        self.idx = 0
        self.c_iter = 0
        self.drop_class = set()

    def shuffle_list(self):
        for c_iter in range(self.class_num):
            np.random.shuffle(self.s_list[c_iter])

    def set_length(self, length: int):
        self.data_len = length

    def set_probmatrix(self, prob_mat):
        self.prob_mat = prob_mat

    def get_list(self):
        self.random_loader = False
        winList = np.argmax(self.prob_mat, axis=1)
        for c_iter in range(self.class_num):
            cur_data = np.where(winList == c_iter)[0]
            self.s_list.append(cur_data)
            cur_length = math.floor((len(cur_data) * self.class_num) / self.batch_num)
            if (cur_length < 1):
                self.drop_class.add(c_iter)
                continue
            if (cur_length < self.data_len):
                self.set_length(cur_length)
                self.i_range = len(cur_data)
        if(len(self.drop_class) > 0):
            cur_length = math.floor((self.i_range * (self.class_num-len(self.drop_class))) / self.batch_num)
            self.set_length(cur_length)
        return True

    def __iter__(self):
        batch = []
        bs = self.batch_num
        if(self.random_loader):
            while(True):
                np.random.shuffle(self.all_data)
                for idx in range(self.i_range):
                    for b_iter in range(bs):
                        batch.append(self.all_data[idx*bs+b_iter])
                    yield batch
                    batch = []
        else:
            batch_ctr = 0
            cur_ctr = 0
            pick_item = np.arange(self.class_num)
            while(True):
                new_round = False
                for idx in range(self.i_range):
                    if(new_round):
                        break
                    np.random.shuffle(pick_item)
                    for c_iter in range(self.class_num):
                        if(new_round):
                            break
                        c_iter_l = pick_item[c_iter]
                        if c_iter_l in self.drop_class:
                            continue
                        c_idx = idx % len(self.s_list[c_iter_l])
                        batch.append(self.s_list[c_iter_l][c_idx])
                        cur_ctr += 1
                        if(cur_ctr % bs == 0):
                            yield batch
                            batch = []
                            cur_ctr = 0
                            batch_ctr += 1
                            if(batch_ctr == self.data_len):
                                batch_ctr = 0
                                self.shuffle_list()
                                new_round = True

    def __len__(self):
        return self.data_len

    def get_range(self):
        return self.i_range