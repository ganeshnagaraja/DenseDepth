import glob
import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
from numpy.ma.core import transpose
from PIL import Image

import torch
from api import utils as api_utils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import cv2


def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}

def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train

class depthDatasetMemory(Dataset):
    def __init__(self, input_type, rgb_dir, labels_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.input_type = input_type
        # Create list of filenames
        self._datalist_rgb = []
        self._datalist_label = []
        self._extension_input = '.jpg'  # The file extension of input images
        self._extension_label = '.exr'  # The file extension of labels
        self._create_lists_filenames(self.rgb_dir, self.labels_dir)

    def __getitem__(self, idx):
        # Open input imgs
        if self.input_type == 'normal':
            image = api_utils.exr_loader(self._datalist_rgb[idx])
            image = (image + 1) / 2.0
            image = image.transpose(1, 2, 0)
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
        elif self.input_type == 'rgb':
            image = Image.open(self._datalist_rgb[idx])

        if self.labels_dir:
            depth = api_utils.exr_loader(self._datalist_label[idx], ndim=1)  # Image.open( BytesIO(self.data[sample[1]]) )
            depth = np.clip(depth, 0.0, 3.0)
            depth = ((depth / 3.0) * 255).astype(np.uint8)
            depth = Image.fromarray(depth)

            sample = {'image': image, 'depth': depth}
            if self.transform: sample = self.transform(sample)
        else:
            print('hello')
            depth = torch.zeros((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)
        return sample

    def __len__(self):
        return len(self._datalist_rgb)

    def _create_lists_filenames(self, rgb_dir, labels_dir):
        assert os.path.isdir(rgb_dir), 'Dataloader given images directory that does not exist: "%s"' % (rgb_dir)
        assert os.path.isdir(labels_dir), 'Dataloader given labels directory that does not exist: "%s"' % (labels_dir)

        # make list of  normals files
        normalsSearchStr = os.path.join(rgb_dir, '*' + self._extension_input)
        normalspaths = sorted(glob.glob(normalsSearchStr))

        self._datalist_rgb = normalspaths

        numImages = len(self._datalist_rgb)
        if labels_dir:
            assert os.path.isdir(labels_dir), ('Dataloader given labels directory that does not exist: "%s"'
                                               % (labels_dir))
            labelSearchStr = os.path.join(labels_dir, '*' + self._extension_label)
            labelpaths = sorted(glob.glob(labelSearchStr))
            numLabels = len(labelpaths)
            if numLabels == 0:
                raise ValueError('No labels found in given directory. Searched for {}'.format(labels_dir))
            if numImages != numLabels:
                raise ValueError('The number of images and labels do not match. Please check data,\
                                found {} images and {} labels' .format(numImages, numLabels))
            self._datalist_label = labelpaths



class ToTensor(object):
    def __init__(self,is_test=False, input_type='rgb'):
        self.is_test = is_test
        self.input_type = input_type

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if self.input_type == 'rgb':
            image = image.resize((640, 480))
        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000

        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test=False, input_type='rgb'):
    return transforms.Compose([
        ToTensor(is_test=is_test, input_type=input_type)
    ])

def getDefaultTrainTransform(input_type='rgb'):
    return transforms.Compose([
        ToTensor(input_type=input_type)
    ])

def getTrainingTestingData(input_type, activity, rgb_dir, labels_dir):

    if activity == 'train':
        transformed_training = depthDatasetMemory(input_type, rgb_dir, labels_dir, transform=getDefaultTrainTransform(input_type=input_type))
        return transformed_training

    elif activity == 'eval':
        transformed_testing = depthDatasetMemory(input_type, rgb_dir, labels_dir, transform=getNoTransform(input_type=input_type))
        return transformed_testing


