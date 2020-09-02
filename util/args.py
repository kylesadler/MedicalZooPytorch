import time
import argparse

def datestr():
    now = time.gmtime()
    return '{:02}-{:02}_{:02}-{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)


def get_default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSz', type=int, default=8)
    parser.add_argument('--nEpochs', type=int, default=20000)

    parser.add_argument('--samples_train', type=int, default=1024)
    parser.add_argument('--samples_val', type=int, default=128)
    # parser.add_argument('--samples_train', type=int, default=1)
    # parser.add_argument('--samples_val', type=int, default=1)

    parser.add_argument('--model', type=str, default='UNET3D', choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))

    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str, default='./runs/')
    parser.add_argument('--cuda', action='store_true', default=True)

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    # parser.add_argument('--augmentation', default=kwargs['augmentation'], type=str, help='Tensor normalization: options max, mean, global')
    parser.add_argument('--normalization', default='full_volume_mean', type=str, help='Tensor normalization: options max, mean, global')

    return parser

def get_parser(**kwargs):
    """ fold_id, loadData, split """
    parser = get_default_parser()

    parser.add_argument('--loadData', default=kwargs['loadData'])
    parser.add_argument('--split', default=kwargs['split'], type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--fold_id', default=str(kwargs['fold_id']), type=str, help='Select subject for fold validation')


    parser.add_argument('--dataset_name', type=str, default=kwargs['dataset_name'])
    parser.add_argument('--dim', nargs="+", type=int, default=kwargs['dim'])
    parser.add_argument('--classes', type=int, default=kwargs['classes'])

    parser.add_argument('--inChannels', type=int, default=kwargs['inChannels'])
    parser.add_argument('--inModalities', type=int, default=kwargs['inModalities'])
    parser.add_argument('--threshold', default=kwargs['threshold'], type=float)

    parser.add_argument('--lr', default=kwargs['lr'], type=float, help='learning rate (default: 1e-3)')

    return parser

""" 
def mrbrains9_arguments(loadData=False):
        loadData=False to recompute data
        loadData=True to use data on disk

    return to_args({
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

def iseg2019_arguments(loadData):

    return to_args({
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

def brats2019_arguments(fold_id, loadData):

    # 80 / 20% train test split. proportional HGG and LGG

    return to_args({
        'fold_id': fold_id,
        'loadData': loadData,
        'split': 0.9,

        'dataset_name': "brats2019",
        'dim': (64, 64, 64),
        'classes': 3, # whole tumor, enhanced tumor, tumor core
        'inChannels': 4,
        'inModalities': 4,
        'threshold': 0.1,
        'lr': 1e-4
    })
 """


def to_args(kwargs):
    """ generates argparse arguments from a keyword configuration dict

    Args:
        kwargs (dict): dictionary of keyword arguments to pass to get_parser()

    Returns:
        argparse arguments: configuration arguments
    """        
    parser = get_parser(**kwargs)
    args = parser.parse_args()
    args.save = f'/home/kyle/results/{args.model}/{args.dataset_name}_{args.fold_id}_{datestr()}'
    return args
