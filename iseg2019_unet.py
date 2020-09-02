import os
import sys
import torch

from pprint import pprint
from util import iseg2019_arguments, launch_train_test, ISEG_UNET_BEST, ISEG_UNET_LAST

import medzoo.lib.medloaders as medical_loaders
import medzoo.lib.medzoo as medzoo
from medzoo.lib.medloaders import MRIDatasetISEG2019, dataset_dir
from medzoo.lib.medzoo import UNet3D
import medzoo.lib.train as train
import medzoo.lib.utils as utilss
from medzoo.lib.losses3D import DiceLoss
from medzoo.lib import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777


args = to_args({
        'loadData': loadData,

        'fold_id': 9,
        'split': 0.8,
        'dataset_name': "iseg2019",
        'dim': (64, 64, 64),
        'classes': 4,
        'inChannels': 2,
        'inModalities': 2,
        'threshold': 0.0001,
        'lr': 1e-3
    })


def train():
    # args = iseg2019_arguments()
    print(args)

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args)

    model, optimizer = medzoo.create_model(args)
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
    # args = iseg2019_arguments()
    print(args)

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)


    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    print(params)
    samples_train = args.samples_train
    samples_val = args.samples_val
    test_loader = MRIDatasetISEG2019(args, 'test', dataset_path=dataset_dir, crop_dim=args.dim, split_id=0,
                                        samples=samples_train, load=args.loadData)
    



    model_name = args.model
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)
    model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    print(model_name, 'Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model.restore_checkpoint("/home/kyle/results/UNET3D/iseg2019_9_06-08_21-25/iseg2019_9_06-08_21-25_BEST.pth")
    criterion = DiceLoss(classes=args.classes)

    # model = model.cuda()
    # print("Model transferred in GPU.....")

    print("TESTING...")

    model.eval()

    confusion_matrix = [ [0]*4 for i in range(4) ]

    for batch_idx, input_tuple in enumerate(test_loader):
        with torch.no_grad():
            img_t1, img_t2, target = input_tuple
            
            target = torch.reshape(target, (-1, 1, 64, 64, 64))
            img_t1 = torch.reshape(img_t1, (-1, 1, 64, 64, 64))
            img_t2 = torch.reshape(img_t2, (-1, 1, 64, 64, 64))

            input_tensor = torch.cat((img_t1, img_t2), dim=1)
            # print(input_tensor.size())

            input_tensor.requires_grad = False

            output = model(input_tensor)

            output = torch.argmax(output, dim=1)
            output = torch.reshape(output, (-1, 1, 64, 64, 64))

            assert target.size() == output.size()

            output = torch.reshape(output, (-1,)).tolist()
            target = torch.reshape(target, (-1,)).tolist()

            assert len(output) == len(target)

            for gt, pred in zip(target, output):
                confusion_matrix[int(gt)][int(pred)] += 1

    pprint(confusion_matrix)



if __name__ == '__main__':
    launch_train_test(train, test, sys.argv)