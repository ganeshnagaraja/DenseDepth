import matplotlib
import matplotlib.cm
import numpy as np
from api import utils as api_utils
import json
import numpy as np
import torch
import torch.nn as nn

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2, 0, 1))

# functions from pytorch-depth utils module
def normalize_depth(depth_img, min_depth=0.0, max_depth=3.0):
    """Normalize a metric depth image to range (-1, 1)

    Args:
        depth_img (numpy.ndarray): A depth image with units of meters.
                                   Shape=(H, W), dtype=np.float32
        min_depth (float, optional): Min depth to be considered in depth image, in meters. Defaults to 0.0.
        max_depth (float, optional): Max depth to be considered in depth image, in meters. Defaults to 3.0.
    """
    # clip depth to given range
    _label = np.clip(depth_img, min_depth, max_depth)

    # normalize depth to range (-1, 1)
    mean_depth = (max_depth + min_depth) / 2
    std_dev_depth = max_depth - mean_depth
    depth_img = (_label - mean_depth) / std_dev_depth

    return depth_img

def get_mask(variant_mask_path, json_path):
    """get a mask of transparent objects in an image from synthetic dataset

    Args:
        variant_mask_path (str): path to variant mask file in  synthetic dataset
        json_path (str): path to json file in synthetic dataset

    Returns:
        numpy.ndarray: mask of transparant objects where pixel value is True where ever transparent object is present
                       dtype = np.bool
    """
    variant_mask = api_utils.exr_loader(variant_mask_path, ndim=1)

    # read json file
    json_file = open(json_path)
    data = json.load(json_file)

    object_id = []
    for key, values in data['variants']['masks_and_poses_by_pixel_value'].items():
        object_id.append(key)

    # create different masks
    final_mask = np.zeros(variant_mask.shape, dtype=np.bool)

    for i in range(len(object_id)):
        mask = np.zeros(variant_mask.shape, dtype=np.bool)
        mask[variant_mask == int(object_id[i])] = True
        final_mask += mask

    return final_mask

def compute_errors(gt, pred):
    """
    Parameters
    ----------
    gt : torch.Tensor
        shape [batch, h, w]
    pred : torch.Tensor
        shape [batch, h, w]

    Return
    ------
    measures : dict
    """
    safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
    safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
    batch_size = pred.shape[0]
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]
    thresh = torch.max(gt / pred, pred / gt)
    a1 = (thresh < 1.25).float().mean() * batch_size
    a2 = (thresh < 1.25 ** 2).float().mean() * batch_size
    a3 = (thresh < 1.25 ** 3).float().mean() * batch_size

    rmse = ((gt - pred) ** 2).mean().sqrt() * batch_size
    rmse_log = ((safe_log(gt) - safe_log(pred))** 2).mean().sqrt() * batch_size
    log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean() * batch_size
    abs_rel = ((gt - pred).abs() / gt).mean() * batch_size
    sq_rel = ((gt - pred)**2 / gt).mean() * batch_size
    measures = {'a1': a1, 'a2': a2, 'a3': a3, 'rmse': rmse,
                'rmse_log': rmse_log, 'log10': log10, 'abs_rel': abs_rel, 'sq_rel': sq_rel}
    return measures