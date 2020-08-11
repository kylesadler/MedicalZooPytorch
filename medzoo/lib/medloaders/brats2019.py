import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import medzoo.lib.augment3D as augment3D
import medzoo.lib.utils as utils
from medzoo.lib.medloaders import medical_image_process as img_loader
from medzoo.lib.medloaders.medical_loader_utils import create_sub_volumes


class MICCAIBraTS2019(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, mode, dataset_path='./datasets', classes=5, crop_dim=(200, 200, 150), split_idx=260,
                 samples=10,
                 load=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param split_idx: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + '/brats2019/MICCAI_BraTS_2019_Data_Training/'
        self.testing_path = self.root + '/brats2019/MICCAI_BraTS_2019_Data_Validation/'
        self.full_vol_dim = (240, 240, 155)  # slice, width, height
        self.crop_size = crop_dim
        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        self.samples = samples
        self.full_volume = None
        self.classes = classes
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)
        self.save_name = self.root + '/brats2019/brats2019-list-' + mode + '-samples-' + str(samples) + '.txt'

        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t1.nii.gz')))
            self.affine = img_loader.load_affine_matrix(list_IDsT1[0])
            return

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
        self.sub_vol_path = self.root + '/brats2019/MICCAI_BraTS_2019_Data_Training/generated/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)



        # split HGG and LGG
        HGG_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, 'HGG/*/*t1.nii.gz')))
        HGG_IDsT1ce = sorted(glob.glob(os.path.join(self.training_path, 'HGG/*/*t1ce.nii.gz')))
        HGG_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, 'HGG/*/*t2.nii.gz')))
        HGG_IDsFlair = sorted(glob.glob(os.path.join(self.training_path, 'HGG/*/*_flair.nii.gz')))
        HGG_labels = sorted(glob.glob(os.path.join(self.training_path, 'HGG/*/*_seg.nii.gz')))

        LGG_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, 'LGG/*/*t1.nii.gz')))
        LGG_IDsT1ce = sorted(glob.glob(os.path.join(self.training_path, 'LGG/*/*t1ce.nii.gz')))
        LGG_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, 'LGG/*/*t2.nii.gz')))
        LGG_IDsFlair = sorted(glob.glob(os.path.join(self.training_path, 'LGG/*/*_flair.nii.gz')))
        LGG_labels = sorted(glob.glob(os.path.join(self.training_path, 'LGG/*/*_seg.nii.gz')))

        HGG_IDsT1, HGG_IDsT1ce, HGG_IDsT2, HGG_IDsFlair, HGG_labels = utils.shuffle_lists(HGG_IDsT1, 
                                                                                            HGG_IDsT1ce,
                                                                                            HGG_IDsT2,
                                                                                            HGG_IDsFlair,
                                                                                            HGG_labels,
                                                                                            seed=17
                                                                                        )

        LGG_IDsT1, LGG_IDsT1ce, LGG_IDsT2, LGG_IDsFlair, LGG_labels = utils.shuffle_lists(LGG_IDsT1, 
                                                                                            LGG_IDsT1ce,
                                                                                            LGG_IDsT2,
                                                                                            LGG_IDsFlair,
                                                                                            LGG_labels,
                                                                                            seed=17
                                                                                        )

        self.affine = img_loader.load_affine_matrix((HGG_IDsT1+LGG_IDsT1)[0])

        hgg_len = len(HGG_IDsT1)
        lgg_len = len(LGG_IDsT1)
        print('Brats2019, Training HGG:', hgg_len)
        print('Brats2019, Training LGG:', lgg_len)
        print('Brats2019, Training total:', hgg_len + lgg_len)

        hgg_split = int(hgg_len * .8)
        lgg_split = int(lgg_len * .8)

        if self.mode == 'train':
            list_IDsT1 = HGG_IDsT1[:hgg_split] + LGG_IDsT1[:hgg_split]
            list_IDsT1ce = HGG_IDsT1ce[:hgg_split] + LGG_IDsT1ce[:hgg_split]
            list_IDsT2 = HGG_IDsT2[:hgg_split] + LGG_IDsT2[:hgg_split]
            list_IDsFlair = HGG_IDsFlair[:hgg_split] + LGG_IDsFlair[:hgg_split]
            labels = HGG_labels[:hgg_split] + LGG_labels[:hgg_split]
            self.list = create_sub_volumes(list_IDsT1, list_IDsT1ce, list_IDsT2, list_IDsFlair, labels,
                                           dataset_name="brats2019", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold)

        elif self.mode == 'val':
            list_IDsT1 = HGG_IDsT1[hgg_split:] + LGG_IDsT1[hgg_split:]
            list_IDsT1ce = HGG_IDsT1ce[hgg_split:] + LGG_IDsT1ce[hgg_split:]
            list_IDsT2 = HGG_IDsT2[hgg_split:] + LGG_IDsT2[hgg_split:]
            list_IDsFlair = HGG_IDsFlair[hgg_split:] + LGG_IDsFlair[hgg_split:]
            labels = HGG_labels[hgg_split:] + LGG_labels[hgg_split:]
            self.list = create_sub_volumes(list_IDsT1, list_IDsT1ce, list_IDsT2, list_IDsFlair, labels,
                                           dataset_name="brats2019", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold)
        elif self.mode == 'test':
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t1.nii.gz')))
            self.list_IDsT1ce = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t1ce.nii.gz')))
            self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t2.nii.gz')))
            self.list_IDsFlair = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*_flair.nii.gz')))
            self.labels = None
            # Todo inference code here

        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        f_t1, f_t1ce, f_t2, f_flair, f_seg = self.list[index]
        img_t1, img_t1ce, img_t2, img_flair, img_seg = np.load(f_t1), np.load(f_t1ce), np.load(f_t2), np.load(
            f_flair), np.load(f_seg)
        if self.mode == 'train' and self.augmentation:
            [img_t1, img_t1ce, img_t2, img_flair], img_seg = self.transform([img_t1, img_t1ce, img_t2, img_flair],
                                                                            img_seg)

            return torch.FloatTensor(img_t1.copy()).unsqueeze(0), torch.FloatTensor(img_t1ce.copy()).unsqueeze(
                0), torch.FloatTensor(img_t2.copy()).unsqueeze(0), torch.FloatTensor(img_flair.copy()).unsqueeze(
                0), torch.FloatTensor(img_seg.copy())

        return img_t1, img_t1ce, img_t2, img_flair, img_seg
