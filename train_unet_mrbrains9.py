# Python medzoo.libraries
import argparse
import os

import torch

import medzoo.lib.medloaders as medical_loaders
import medzoo.lib.medzoo as medzoo
import medzoo.lib.train as train
# medzoo.lib files
import medzoo.lib.utils as utils
from medzoo.lib.losses3D.dice import DiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 1777777
torch.manual_seed(seed)


def main():
    args = get_arguments()

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args)
    model, optimizer = medzoo.create_model(args)
    criterion = DiceLoss(classes=11, skip_index_after=args.classes)

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=8)
    parser.add_argument('--dataset_name', type=str, default="mrbrains9")
    # max size is (240, 240, 48)
    parser.add_argument('--dim', nargs="+", type=int, default=(48, 48, 48))
    parser.add_argument('--nEpochs', type=int, default=20000)
    parser.add_argument('--classes', type=int, default=9)

    
    # parser.add_argument('--samples_train', type=int, default=4096)
    # parser.add_argument('--samples_val', type=int, default=512)
    parser.add_argument('--samples_train', type=int, default=1024)
    parser.add_argument('--samples_val', type=int, default=128)
    # parser.add_argument('--samples_train', type=int, default=1)
    # parser.add_argument('--samples_val', type=int, default=1)

    # parser.add_argument('--inChannels', type=int, default=3)
    # parser.add_argument('--inModalities', type=int, default=3)
    parser.add_argument('--inChannels', type=int, default=2)
    parser.add_argument('--inModalities', type=int, default=2)
    parser.add_argument('--threshold', default=0.1, type=float)
    parser.add_argument('--augmentation', default='no', type=str,
                        help='Tensor normalization: options max, mean, global')
    parser.add_argument('--normalization', default='global_mean', type=str,
                        help='Tensor normalization: options max, mean, global')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--split', default=0.9, type=float, help='Select percentage of training data(default: 0.8)')

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')

    parser.add_argument('--cuda', action='store_true', default=True)

    # parser.add_argument('--loadData', default=True)
    parser.add_argument('--loadData', default=True)

    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='./runs/')

    args = parser.parse_args()

    args.save = f'/home/kyle/results/{args.model}/{args.dataset_name}_{args.fold_id}_{utils.datestr()}'

    return args


if __name__ == '__main__':
    main()
