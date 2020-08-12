# Python medzoo.libraries
import argparse
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# medzoo.lib files
from medzoo.lib.medloaders import MRIDatasetMRBRAINS2018, dataset_dir
from medzoo.lib.medzoo import UNet3D
import medzoo.lib.train as train
import medzoo.lib.utils as utilss
from medzoo.lib.losses3D import DiceLoss
from medzoo.lib import utils

from pprint import pprint

seed = 1777777


def main():
    args = get_arguments()
    print(args)

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

    model.eval()

    confusion_matrix = [ [0]*num_classes for i in range(num_classes) ]

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
    parser.add_argument('--samples_train', type=int, default=1)
    parser.add_argument('--samples_val', type=int, default=128)
    # parser.add_argument('--samples_train', type=int, default=1)
    # parser.add_argument('--samples_val', type=int, default=1)

    parser.add_argument('--inChannels', type=int, default=3)
    parser.add_argument('--inModalities', type=int, default=3)
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
    parser.add_argument('--loadData', default=False)

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
