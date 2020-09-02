import os
import sys
import torch
from util import to_args, launch_train_test, BRATS_UNET_BEST, BRATS_UNET_LAST

import medzoo.lib.medloaders as medical_loaders
import medzoo.lib.medzoo as medzoo
import medzoo.lib.train as train
# medzoo.lib files
import medzoo.lib.utils as utils
from medzoo.lib.losses3D.dice import DiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = 1777777

args = to_args({
        'fold_id': '1', # train / test split happens in brats2019.py
        'loadData': False,
        'split': 0.9,

        'dataset_name': "brats2019",
        'dim': (64, 64, 64),
        'classes': 3, # whole tumor, enhanced tumor, tumor core
        'inChannels': 4,
        'inModalities': 4,
        'threshold': 0.1,
        'lr': 1e-4
        })

def train():
    # args = brats2019_arguments()

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args)
    model, optimizer = medzoo.create_model(args)
    # criterion = DiceLoss(classes=3, skip_index_after=args.classes)
    # criterion = DiceLoss(classes=args.classes)
    criterion = torch.nn.CrossEntropyLoss()


    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()



def test():
    raise NotImplementedError


if __name__ == '__main__':
    launch_train_test(train, test, sys.argv)