import json
import os
import random
import shutil
import time
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def reproducibility(args, seed):
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # FOR FASTER GPU TRAINING WHEN INPUT SIZE DOESN'T VARY
    # LET'S TEST IT
    cudnn.benchmark = True


def save_arguments(args, path):
    with open(path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()


def datestr():
    now = time.gmtime()
    return '{:02}-{:02}_{:02}-{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)


def shuffle_lists(*ls, seed=777):
    l = list(zip(*ls))
    random.seed(seed)
    random.shuffle(l)
    return zip(*l)


def prepare_input(input_tuple, inModalities=-1, inChannels=-1, cuda=False, args=None):
    if args is not None:
        modalities = args.inModalities
        channels = args.inChannels
        in_cuda = args.cuda
    else:
        modalities = inModalities
        channels = inChannels
        in_cuda = cuda
    if modalities == 4:
        if channels == 4:
            img_1, img_2, img_3, img_4, target = input_tuple
            # size of each is 8 x 64 x 64 x 64
            # they get concatenated into 8 x 256 x 64 x 64
            # but model wants 8 x 3 x 64 x 64 x 64
            if args.dataset_name == 'brats2019':
                img_1 = torch.reshape(img_1, (8, 1, 64, 64, 64))
                img_2 = torch.reshape(img_2, (8, 1, 64, 64, 64))
                img_3 = torch.reshape(img_3, (8, 1, 64, 64, 64))
                img_4 = torch.reshape(img_4, (8, 1, 64, 64, 64))
                
            input_tensor = torch.cat((img_1, img_2, img_3, img_4), dim=1)
        elif channels == 3:
            # t1 post constast is ommited
            img_1, _, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_3, img_4), dim=1)
        elif channels == 2:
            # t1 and t2 only
            img_1, _, img_3, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_3), dim=1)
        elif channels == 1:
            # t1 only
            input_tensor, _, _, target = input_tuple
    if modalities == 3:
        if channels == 3:
            img_1, img_2, img_3, target = input_tuple
            # size of each is 8 x 48 x 48 x 48
            # they get concatenated into 8 x 144 x 48 x 48
            # but model wants 8 x 3 x 48 x 48 x 48
            if args.dataset_name == 'mrbrains9':
                img_1 = torch.reshape(img_1, (8, 1, 48, 48, 48))
                img_2 = torch.reshape(img_2, (8, 1, 48, 48, 48))
                img_3 = torch.reshape(img_3, (8, 1, 48, 48, 48))

            input_tensor = torch.cat((img_1, img_2, img_3), dim=1)

        elif channels == 2:
            img_1, img_2, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_2), dim=1)
        elif channels == 1:
            input_tensor, _, _, target = input_tuple
    elif modalities == 2:
        if channels == 2:
            img_t1, img_t2, target = input_tuple
            if args.dataset_name == 'iseg2019':
                print(img_t1.size())
                print(img_t2.size())
                img_t1 = torch.reshape(img_t1, (-1, 1, 64, 64, 64))
                img_t2 = torch.reshape(img_t2, (-1, 1, 64, 64, 64))
                print(img_t1.size())
                print(img_t2.size())

            input_tensor = torch.cat((img_t1, img_t2), dim=1)
            print(input_tensor.size())

        elif channels == 1:
            input_tensor, _, target = input_tuple
    elif modalities == 1:
        input_tensor, target = input_tuple

    if in_cuda:
        input_tensor, target = input_tensor.cuda(), target.cuda()

    return input_tensor, target


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def save_list(name, list):
    with open(name, "wb") as fp:
        pickle.dump(list, fp)


def load_list(name):
    with open(name, "rb") as fp:
        list_file = pickle.load(fp)
    return list_file
