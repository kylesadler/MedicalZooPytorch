from .paths import MRBRAINS_UNET_BEST, MRBRAINS_UNET_LAST, ISEG_UNET_BEST, ISEG_UNET_LAST, BRATS_UNET_BEST, BRATS_UNET_LAST

from .args import to_args

def launch_train_test(train, test, sysargs):
    if sysargs[1] == 'train':
        train()
    elif sysargs[1] == 'test':
        test()
    else:
        raise ValueError('first argument must be train or test')

