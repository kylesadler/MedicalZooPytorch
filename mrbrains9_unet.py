import os
import sys
import torch
from pprint import pprint

from util import mrbrains9_arguments, launch_train_test, MRBRAINS_UNET_BEST, MRBRAINS_UNET_LAST

# Python medzoo.libraries
# medzoo.lib files
from medzoo.lib.medzoo import UNet3D
from medzoo.lib.losses3D.dice import DiceLoss
from medzoo.lib.medloaders import MRIDatasetMRBRAINS2018, dataset_dir

import medzoo.lib.train as train
import medzoo.lib.utils as utils
import medzoo.lib.medzoo as medzoo
import medzoo.lib.medloaders as medical_loaders

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777

# from medzoo.lib import utils

args = to_args({
        'loadData': loadData,

        'fold_id': 148,
        'split': 0.9,
        'dataset_name': "mrbrains9",
        'dim': (48, 48, 48), # max size is (240, 240, 48)
        'classes': 9,
        'inChannels': 3,
        'inModalities': 3,
        'threshold': 0.1,
        'lr': 1e-4
    })


def train():
    # args = mrbrains9_arguments(loadData=True)
    print('here')
    pprint(args)
    print('there')

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args)
    model, optimizer = medzoo.create_model(args)
    # criterion = DiceLoss(classes=11, skip_index_after=args.classes)
    # criterion = DiceLoss(classes=args.classes)
    criterion = CrossEntropyLoss()

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()


def test():
    # args = mrbrains9_arguments(loadData=True)

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)


    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    samples_train = args.samples_train
    samples_val = args.samples_val
    test_loader = MRIDatasetMRBRAINS2018(args, 'test', dataset_path=dataset_dir, dim=args.dim, split_id=0,
                                        samples=samples_train, load=args.loadData)
    

    model_name = args.model
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)
    model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    print(model_name, 'Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model.restore_checkpoint("/home/kyle/results/UNET3D/mrbrains9_148_09-08_17-46/mrbrains9_148_09-08_17-46_BEST.pth")

    # model = model.cuda()
    # print("Model transferred in GPU.....")

    print("TESTING...")


# [[37507023, 290552, 0, 25074, 30, 1323040, 134, 20823, 10884, 0],
#  [256417, 16475613, 1592518, 1920243, 2259, 2491037, 67078, 61665, 497, 0],
#  [60651, 3655997, 594069, 16095428, 2494541, 102916, 456685, 4495, 4150, 0],
#  [1225472, 1183528, 24771, 74524, 614, 13215040, 2649646, 104672, 72727, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [91439, 131011, 492, 8209, 0, 784077, 1, 6204313, 249132, 0],
#  [9185, 3047, 0, 13415, 0, 62690, 0, 12931, 1513371, 0],
#  [1471, 46677, 0, 5422, 10476, 32999, 1025, 12, 0, 0],

    model.eval()

    confusion_matrix = [ [0]*(num_classes*2) for i in range(num_classes*2) ]

    for batch_idx, input_tuple in enumerate(test_loader):
        with torch.no_grad():
            img_t1, img_t2, img_t3, target = input_tuple
            
            target = torch.reshape(torch.from_numpy(target), (-1, 1, 48, 48, 48))
            img_t1 = torch.reshape(torch.from_numpy(img_t1), (-1, 1, 48, 48, 48))
            img_t2 = torch.reshape(torch.from_numpy(img_t2), (-1, 1, 48, 48, 48))
            img_t3 = torch.reshape(torch.from_numpy(img_t3), (-1, 1, 48, 48, 48))

            input_tensor = torch.cat((img_t1, img_t2, img_t3), dim=1)

            input_tensor.requires_grad = False

            output = model(input_tensor)

            output = torch.argmax(output, dim=1)
            output = torch.reshape(output, (-1, 1, 48, 48, 48))

            assert target.size() == output.size()

            output = torch.reshape(output, (-1,)).tolist()
            target = torch.reshape(target, (-1,)).tolist()

            assert  len(output) == len(target)

            for gt, pred in zip(target, output):
                confusion_matrix[int(gt)][int(pred)] += 1

    pprint(confusion_matrix)

if __name__ == '__main__':
    launch_train_test(train, test, sys.argv)
