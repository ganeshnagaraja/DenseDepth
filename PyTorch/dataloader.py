#!/usr/bin/env python3

from __future__ import division, print_function

import glob
import os
import sys

import imageio
import numpy as np
from matplotlib.pyplot import axis
from PIL import Image

import Imath
import imgaug as ia
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from torchvision import transforms
import utils


from api import utils as api_utils

class DepthDataset(Dataset):
    """
    Dataset class for training model on estimation of depth.
    Uses imgaug for image augmentations.

    ONLY works on synthetic dataset images.

    Args:
        normals (str): Path to folder containing the surface normals (.exr format).
        variant_masks (str): Path to folder containing the variant masks (.exr format).
        json (str): Path to folder containing the JSON files of synthetic dataset.
        labels(str): Path to folder containing the rectified depth images (.exr format).
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img
        min_depth (float): Min depth to which depth image is clipped.
        max_depth (float): Max depth to which depth image is clipped.
        concat_depth (bool): Default True. Whether to concat the input depth with holes to the normals
        normalize_depth (bool): Default True. Whether to normalize the depth images to range (-1, 1)
    """

    def __init__(self,
                 normals='data/datasets/train/milk-bottles-train/source-files/camera-normals',
                 variant_masks='data/datasets/train/milk-bottles-train/source-files/variant-masks',
                 json='data/datasets/train/milk-bottles-train/source-files/json-files',
                 labels='data/datasets/train/milk-bottles-train/source-files/depth-imgs-rectified',
                 transform=None,
                 input_only=None,
                 min_depth=0.0,
                 max_depth=3.0,
                 concat_depth=True,
                 model='rednet',
                 type_of_data_loaded='train'
                 ):

        super().__init__()

        self.normals_dir = normals
        self.variant_masks_dir = variant_masks
        self.json_dir = json
        self.labels_dir = labels
        self.transform = transform
        self.input_only = input_only
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.concat_depth = concat_depth
        self.model = model
        self.type_of_data_loaded = type_of_data_loaded

        # Create list of filenames
        self._datalist_normals = []  # variable containing list of all surface normals ground truth filenames
        self._datalist_variants = []  # variable containing list of all variant masks filenames
        self._datalist_json = []  # variable containing list of all json filenames
        self._datalist_label = []  # Variable containing list of all ground truth filenames in dataset
        self._extension_input = '.exr'  # The file extension of input images
        self._extension_label = '.exr'  # The file extension of labels
        self._create_lists_filenames(self.normals_dir, self.variant_masks_dir, self.json_dir, self.labels_dir)

    def __len__(self):
        return len(self._datalist_normals)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label
        '''
        # Open input imgs
        normals = api_utils.exr_loader(self._datalist_normals[index])

        # Open labels
        if self.labels_dir:
            label_path = self._datalist_label[index]
            _label = api_utils.exr_loader(label_path, ndim=1)
            _label[np.isinf(_label)] = self.max_depth
            _label[np.isnan(_label)] = 0.0
            mask = utils.get_mask(self._datalist_variants[index], self._datalist_json[index])
            depth_with_mask = _label.copy()
            depth_with_mask[mask] = 0.0


        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            normals = det_tf.augment_image(normals.transpose(1, 2, 0))
            depth_with_mask = det_tf.augment_image(depth_with_mask)
            depth_with_mask = depth_with_mask[..., np.newaxis]

            # whether to contact depth with normals as input or not
            if self.concat_depth and self.model != 'rednet':
                _img = np.concatenate((normals, depth_with_mask), axis=2)
            else:
                _img = normals

            if self.labels_dir:
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))
                _label = _label[np.newaxis, ...]

        # Return Tensors
        _img_tensor = torch.from_numpy(_img.transpose(2,0,1))
        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label)
            if self.type_of_data_loaded == 'train':
                _label_tensor = _label_tensor.float() * 1000
            elif self.type_of_data_loaded == 'test':
                _label_tensor = _label_tensor.float() / 1000
        else:
            _label_tensor = torch.zeros((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        # convert mask to tensor
        mask = det_tf.augment_image(mask)
        mask = mask[np.newaxis, ...]
        mask = torch.from_numpy(mask.astype(np.uint8))
        mask = mask.byte()

        #convert depth with mask to tensor
        depth_with_mask = torch.from_numpy(depth_with_mask.transpose(2, 0, 1))

        if self.model == 'rednet':
            return _img_tensor, depth_with_mask, mask, _label_tensor
        else:
            return _img_tensor, mask, _label_tensor

    def _create_lists_filenames(self, normals_dir, variant_masks_dir, json_dir, labels_dir):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''

        assert os.path.isdir(normals_dir), 'Dataloader given images directory that does not exist: "%s"' % (normals_dir)
        assert os.path.isdir(variant_masks_dir), 'Dataloader given images directory that does not exist: "%s"' % (variant_masks_dir)
        assert os.path.isdir(json_dir), 'Dataloader given images directory that does not exist: "%s"' % (json_dir)
        assert os.path.isdir(labels_dir), 'Dataloader given labels directory that does not exist: "%s"' % (labels_dir)

        # make list of  normals files
        normalsSearchStr = os.path.join(normals_dir, '*' + self._extension_input)
        normalspaths = sorted(glob.glob(normalsSearchStr))

        # make list of variant masks
        variantSearchStr = os.path.join(variant_masks_dir, '*' + self._extension_input)
        variantpaths = sorted(glob.glob(variantSearchStr))

        # make list of json files
        jsonSearchStr = os.path.join(json_dir, '*' + '.json')
        jsonpaths = sorted(glob.glob(jsonSearchStr))

        if len(normalspaths) != len(variantpaths) != len(jsonpaths) != len(depthpaths):
            raise ValueError('length of datasets are not equal, please check')

        self._datalist_normals = normalspaths
        self._datalist_variants = variantpaths
        self._datalist_json = jsonpaths

        numImages = len(self._datalist_normals)
        if numImages == 0:
            raise ValueError('No images found in given directory. Searched for {}'.format(normalspaths))

        if labels_dir:
            assert os.path.isdir(labels_dir), ('Dataloader given labels directory that does not exist: "%s"'
                                               % (labels_dir))
            labelSearchStr = os.path.join(labels_dir, '*' + self._extension_label)
            labelpaths = sorted(glob.glob(labelSearchStr))
            numLabels = len(labelpaths)
            if numLabels == 0:
                raise ValueError('No labels found in given directory. Searched for {}'.format(depthpaths))
            if numImages != numLabels:
                raise ValueError('The number of images and labels do not match. Please check data,\
                                found {} images and {} labels' .format(numImages, numLabels))
            self._datalist_label = labelpaths

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    # imsize = 512
    # augs_train = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0), # Resize image
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Rot90((0, 4)),
    #     # Blur and Noise
    #     #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    #     #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    #     iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
    #     # Color, Contrast, etc.
    #     #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    #     iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    #     iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    #     #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    # ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])

    augs = None  # augs_train, augs_test, None
    input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = DepthDataset(
        input_dir_1='data/datasets/milk-bottles/resized-files/preprocessed-rgb-imgs',
        input_dir_2='',
        label_dir='data/datasets/milk-bottles/resized-files/preprocessed-camera-normals',
        transform=augs,
        input_only=input_only
    )

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, label = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)

        # Show Batch
        sample = torch.cat((img, label), 2)
        im_vis = torchvision.utils.make_grid(sample, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1, 2, 0))
        plt.show()

        break
