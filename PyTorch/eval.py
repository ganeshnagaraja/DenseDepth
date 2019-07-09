'''Inference to predict depth using deeplab resent
'''

import argparse
import glob
import io
import math
import os
import sys

from attrdict import AttrDict
import cv2
import h5py
import imageio
import imgaug as ia
import numpy as np
import torch
import torch.nn as nn
import yaml
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from termcolor import colored
from tqdm import tqdm

from data import getTrainingTestingData
from model import Model
from utils import AverageMeter, DepthNorm, colorize
from loss import ssim

sys.path.append('..')
from api import utils as api_utils


###################### Load Config File #############################
parser = argparse.ArgumentParser(description='Run eval of depth prediction model')
parser.add_argument('-c', '--configFile', required=True, help='Path to yaml config file', metavar='path/to/config.yaml')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = AttrDict(config_yaml)

print('Inference of depth model. Loading checkpoint...')

###################### Load Checkpoint and its data #############################
if not os.path.isfile(config.eval.pathWeightsFile):
    raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
        config.eval.pathWeightsFile))

# Read config file stored in the model checkpoint to re-use it's params
CHECKPOINT = torch.load(config.eval.pathWeightsFile, map_location='cpu')
if 'model_state_dict' in CHECKPOINT:
    print(colored('Loaded data from checkpoint {}'.format(config.eval.pathWeightsFile), 'green'))

    config_checkpoint_dict = CHECKPOINT['config']
    config_checkpoint = AttrDict(config_checkpoint_dict)
    print('    Config from Checkpoint:\n', config_checkpoint_dict, '\n\n')
else:
    raise ValueError('The checkpoint file does not have model_state_dict in it.\
                     Please use the newer checkpoint files!')

# Check for results store dir
DIR_RESULTS_REAL = os.path.join(config.eval.resultsDirReal)
DIR_RESULTS_SYNTHETIC = os.path.join(config.eval.resultsDirSynthetic)

if not os.path.isdir(DIR_RESULTS_REAL):
    print(colored('The dir to store results "{}" does not exist. Creating dir'.format(DIR_RESULTS_REAL), 'red'))
    os.makedirs(DIR_RESULTS_REAL)
if not os.path.isdir(DIR_RESULTS_SYNTHETIC):
    print(colored('The dir to store results "{}" does not exist. Creating dir'.format(DIR_RESULTS_SYNTHETIC), 'red'))
    os.makedirs(DIR_RESULTS_SYNTHETIC)

###################### DataLoader #############################
# Make new dataloaders for each synthetic/real dataset

######DenseNet###########
db_test_list_synthetic = []
if config.eval.datasetsSynthetic is not None:
    for dataset in config.eval.datasetsSynthetic:
        if dataset.images:
            test_data = getTrainingTestingData('rgb', 'train', dataset.images, dataset.labels)
            db_test_list_synthetic.append(test_data)

    test_loader_synthetic = DataLoader(torch.utils.data.ConcatDataset(db_test_list_synthetic), config.eval.batchSize, num_workers=config.eval.numWorkers, shuffle=False, drop_last=False, pin_memory=True)

db_test_list_real = []
if config.eval.datasetsReal is not None:
    for dataset in config.eval.datasetsReal:
        if dataset.images:
            test_data = getTrainingTestingData('rgb', 'train', dataset.images, dataset.labels)
            db_test_list_real.append(test_data)

    test_loader_real = DataLoader(torch.utils.data.ConcatDataset(db_test_list_real), config.eval.batchSize, num_workers=config.eval.numWorkers, shuffle=False, drop_last=False, pin_memory=True)

if len(db_test_list_synthetic) + len(db_test_list_real) == 0:
    raise ValueError('No valid datasets provided to run inference on!')


###################### ModelBuilder #############################
if config.eval.model == 'deeplab_xception':
    model = deeplab.DeepLab(num_classes=config_checkpoint.train.numClasses,
                            backbone='xception',
                            output_stride=config_checkpoint.train.outputStride,
                            sync_bn=True,
                            freeze_bn=False)
elif config.eval.model == 'deeplab_resnet':
    model = deeplab.DeepLab(num_input_channels=config.eval.numInputChannels,
                    num_classes=config.eval.numClasses,
                    backbone='resnet',
                    output_stride=config.eval.outputStride,
                    sync_bn=None,
                    freeze_bn=False)
elif config.eval.model == 'rednet':
    model = RedNet(num_classes=1, pretrained=False)
elif config.eval.model == 'densenet':
    model = Model()
else:
    raise ValueError('Invalid model "{}" in config file. Must be one of ["deeplab_xception", "deeplab_resnet"]'.format(
        config.eval.model))

model.load_state_dict(CHECKPOINT['model_state_dict'])

# Enable Multi-GPU training
print("Let's use", torch.cuda.device_count(), "GPUs!")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

### Run Validation and Test Set ###
print('\nInference - Depth of Transparent Objects')
print('-' * 50 + '\n')
print('Running inference on Test sets at:\n    {}\n    {}\n'.format(config.eval.datasetsReal,
                                                                    config.eval.datasetsSynthetic))
print('Results will be saved to:\n    {}\n    {}\n'.format(config.eval.resultsDirReal, config.eval.resultsDirSynthetic))

dataloaders_dict = {}
if db_test_list_real:
    dataloaders_dict.update({'real': test_loader_real})
if db_test_list_synthetic:
    print('WARNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN')
    dataloaders_dict.update({'synthetic': test_loader_synthetic})

for key in dataloaders_dict:
    print('\n' + key + ':')
    print('=' * 30)

    testLoader = dataloaders_dict[key]
    running_loss = 0.0

    # loss
    l1_criterion = nn.L1Loss()

    for ii, sample_batched in enumerate(tqdm(testLoader)):

        # Forward pass of the mini-batch
        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        # Normalize depth
        depth_n = DepthNorm(depth)

        # Predict
        outputs = model(image)

        # Compute the loss
        l_depth = l1_criterion(outputs, depth_n)
        l_ssim = torch.clamp((1 - ssim(outputs, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

        loss = (1.0 * l_ssim) + (0.1 * l_depth)

        running_loss += loss.item()

        # Save output images, one at a time, to results
        inputs_tensor = image.detach().cpu()
        output_tensor = outputs.detach().cpu()
        label_tensor = depth_n.detach().cpu()

        # Extract each tensor within batch and save results
        for iii, sample_batched in enumerate(zip(inputs_tensor, output_tensor, label_tensor)):
            input, output, label = sample_batched

            if key == 'real':
                RESULTS_DIR = config.eval.resultsDirReal
            else:
                RESULTS_DIR = config.eval.resultsDirSynthetic

            result_path = os.path.join(RESULTS_DIR,
                                       '{:09d}-result.png'.format(ii * config.eval.batchSize + iii))

            # Save Results
            # grid image with input, prediction and label
            output = output.squeeze(0).numpy()
            label = label.squeeze(0).numpy()

            # RGB Image
            input = (input.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # resizing output and normals
            output = cv2.resize(output, (640, 480), interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (640, 480), interpolation=cv2.INTER_NEAREST)


            print('label max/min1:', label.max(), label.min())
            print('output max/min1:', output.max(), output.min())

            output = np.clip(output, 1, 100)
            output = DepthNorm(output)  # De-Normalizing depth
            label = DepthNorm(label)  # De-Normalizing depth

            print('label max/min2:', label.max(), label.min())
            print('output max/min2:', output.max(), output.min())

            output = (output / 1000) * 3.0
            label = (label / 1000) * 3.0

            print('label max/min3:', label.max(), label.min())
            print('output max/min3:', output.max(), output.min())

            output_rgb = api_utils.depth2rgb(output, config.train.min_depth,
                                                        config.train.max_depth)
            label_rgb = api_utils.depth2rgb(label, config.train.min_depth, config.train.max_depth)


            numpy_grid = np.concatenate((input, output_rgb, label_rgb), axis=1)
            imageio.imwrite(result_path, numpy_grid)

            # Save Point Cloud
            ptcloud_path = os.path.join(RESULTS_DIR,
                                        '{:09d}-output-ptcloud.ply'.format(ii * config.eval.batchSize + iii))
            print('img {:09d} min-depth {} max-depth {}'. format(ii * config.eval.batchSize + iii, output.min(), output.max() ))

            fov_x = 69.4  # degrees
            fov_y = 42.56  # degrees
            h, w = input.shape[0], input.shape[1]
            cx = w / 2
            cy = h / 2
            fx = cx / (math.tan(math.radians(fov_x / 2)))
            fy = cy / (math.tan(math.radians(fov_y / 2)))

            api_utils.write_point_cloud(ptcloud_path, input, output, fx, fy, cx, cy)

    epoch_loss = running_loss / (len(testLoader))
    print('\nTest Mean Loss ({}): {:.4f}'.format(config_checkpoint.train.loss_type, epoch_loss))
