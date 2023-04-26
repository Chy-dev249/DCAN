import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


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


def get_class_state(lines):
    overall_class_stats = {}
    samples_with_class = {}
    for line in lines:
        path = line.split(' ')[0]
        c = int(line.split(' ')[1].split('\n')[0])
        
        if c not in overall_class_stats:
            overall_class_stats[c] = 1
            samples_with_class[c] = []
        else:
            overall_class_stats[c] += 1
        samples_with_class[c].append(path)
    overall_class_stats = {k: v
                            for k, v in sorted(
                            overall_class_stats.items(), key=lambda item: item[0])}
    class_freq = torch.tensor(list(overall_class_stats.values()))
    return class_freq, samples_with_class, list(overall_class_stats.keys())
    
    
def get_class_probs(class_freq, temperature):
    return torch.softmax(class_freq / temperature, dim=-1) # .numpy()
            

def get_class_temp_linear(max_temp, min_temp, max_iter, current_iter):
    temp = max_temp - (max_temp-min_temp) / max_iter * current_iter
    return temp


class SCPList(Dataset):
    ''' Sample by class probability '''
    def __init__(self, image_list, labels=None, mode='RGB', dynamic_p=True,
                 transform=None, target_transform=None, max_iter=20000, max_temp = 1, min_temp = 0.01, batchsize=36):
        # repeat dataset for setting drop_last=True
        total_data_num = max_iter*batchsize
        self.imgs = make_dataset(image_list, labels)
        repeat_factor = total_data_num//len(self.imgs) + 1
        self.imgs = self.imgs * repeat_factor
        
        self.class_freq, self.class_samples, self.class_keys = get_class_state(image_list)
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.image_list = image_list
        self.max_iter = max_iter
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.dynamic_p = dynamic_p
        self.current_iter = 0
        
    def __getitem__(self, idx):
        if self.dynamic_p:
            class_temp = get_class_temp_linear(self.max_temp, self.min_temp,
                                               self.max_iter, self.current_iter)
        else:
            class_temp = self.max_temp
        class_probs = get_class_probs(self.class_freq, class_temp)
        target = int(torch.multinomial(class_probs, 1,replacement=True))
        index = int(torch.randint(len(self.class_samples[target]),(1,1)))
        path = self.class_samples[target][index]
        
        # numpy.random.choice always return same number for multi-process
        # target = np.random.choice(self.class_keys, p=class_probs)
        # path = np.random.choice(self.class_samples[target])
        
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
     
    def update_current_iter(self):
        self.current_iter += 1

    def __len__(self):
        return len(self.imgs)


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', is_train=False, batchsize=36, max_iter=20000):
        # repeat dataset for setting drop_last=True
        self.imgs = make_dataset(image_list, labels)
        if is_train:
            total_data_num = max_iter*batchsize
            repeat_factor = total_data_num//len(self.imgs) + 1
            self.imgs = self.imgs * repeat_factor

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