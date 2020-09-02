import os

mrbrains_name = 'mrbrains9_148_09-08_17-46'
iseg_name = 'iseg2019_9_06-08_21-25'
brats_name = 'brats2019_1_11-08_04-13'

util_dir = os.path.dirname(__file__)
path_prefix = os.path.join(util_dir, '..', 'results', 'UNET3D')

mrbrains_path = os.path.join(path_prefix, mrbrains_name, mrbrains_name)
iseg_path = os.path.join(path_prefix, iseg_name, iseg_name)
brats_path = os.path.join(path_prefix, brats_name, brats_name)


MRBRAINS_UNET_BEST = f'{mrbrains_path}_BEST.pth'
MRBRAINS_UNET_LAST = f'{mrbrains_path}_last_epoch.pth'

ISEG_UNET_BEST = f'{iseg_path}_BEST.pth'
ISEG_UNET_LAST = f'{iseg_path}_last_epoch.pth'

BRATS_UNET_BEST = f'{brats_path}_BEST.pth'
BRATS_UNET_LAST = f'{brats_path}_last_epoch.pth'