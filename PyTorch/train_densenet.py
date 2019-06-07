import argparse
import datetime
import glob
import io
import os
import shutil
import time

import oyaml
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
from attrdict import AttrDict
from data import getTrainingTestingData
from loss import ssim
from model import Model
from tensorboardX import SummaryWriter
from termcolor import colored
from utils import AverageMeter, DepthNorm, colorize


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
    args = parser.parse_args()

    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = oyaml.load(fd)  # Returns an ordered dict. Used for printing

    config = AttrDict(config_yaml)
    print(colored('Config being used for training:\n{}\n\n'.format(oyaml.dump(config_yaml)), 'green'))

    # Create a new directory to save logs
    runs = sorted(glob.glob(os.path.join(config.train.logsDir, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    MODEL_LOG_DIR = os.path.join(config.train.logsDir, 'exp-{:03d}'.format(prev_run_id + 1))
    CHECKPOINT_DIR = os.path.join(MODEL_LOG_DIR, 'checkpoints')
    os.makedirs(CHECKPOINT_DIR)
    print('Saving logs to folder: ' + colored('"{}"'.format(MODEL_LOG_DIR), 'blue'))

    # Save a copy of config file in the logs
    shutil.copy(CONFIG_FILE_PATH, os.path.join(MODEL_LOG_DIR, 'config.yaml'))

    # Create a tensorboard object and Write config to tensorboard
    writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

    string_out = io.StringIO()
    oyaml.dump(config_yaml, string_out, default_flow_style=False)
    config_str = string_out.getvalue().split('\n')
    string = ''
    for line in config_str:
        string = string + '    ' + line + '\n\r'
    writer.add_text('Config', string, global_step=None)

    # Create model
    model = Model()
    print('Model created.')

    # to continue training from a checkpoint
    if config.train.continueTraining:
        print('Transfer Learning enabled. Model State to be loaded from a prev checkpoint...')
        if not os.path.isfile(config.train.pathPrevCheckpoint):
            raise ValueError('Invalid path to the given weights file for transfer learning.\
                    The file {} does not exist'.format(config.train.pathPrevCheckpoint))

        CHECKPOINT = torch.load(config.train.pathPrevCheckpoint, map_location='cpu')

        if 'model_state_dict' in CHECKPOINT:
            # Newer weights file with various dicts
            print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
            print('Config Used to train Checkpoint:\n', oyaml.dump(CHECKPOINT['config']), '\n')
            print('From Checkpoint: Last Epoch Loss:', CHECKPOINT['epoch_loss'], '\n\n')

            model.load_state_dict(CHECKPOINT['model_state_dict'])
        elif 'state_dict' in CHECKPOINT:
            # reading original authors checkpoints
            if config.train.model != 'rednet':
                # original author deeplab checkpoint
                CHECKPOINT['state_dict'].pop('decoder.last_conv.8.weight')
                CHECKPOINT['state_dict'].pop('decoder.last_conv.8.bias')
            else:
                # rednet checkpoint
                # print(CHECKPOINT['state_dict'].keys())
                CHECKPOINT['state_dict'].pop('final_deconv.weight')
                CHECKPOINT['state_dict'].pop('final_deconv.bias')
                CHECKPOINT['state_dict'].pop('out5_conv.weight')
                CHECKPOINT['state_dict'].pop('out5_conv.bias')
                CHECKPOINT['state_dict'].pop('out4_conv.weight')
                CHECKPOINT['state_dict'].pop('out4_conv.bias')
                CHECKPOINT['state_dict'].pop('out3_conv.weight')
                CHECKPOINT['state_dict'].pop('out3_conv.bias')
                CHECKPOINT['state_dict'].pop('out2_conv.weight')
                CHECKPOINT['state_dict'].pop('out2_conv.bias')

            model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
        else:
            # Old checkpoint containing only model's state_dict()
            model.load_state_dict(CHECKPOINT)

    # Enable Multi-GPU training
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        print('Multiple GPUs being used, can\'t save model graph to Tensorboard')
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), config.train.optimAdam.learningRate )
    batch_size = config.train.batchSize
    prefix = 'densenet_' + str(batch_size)

    # Load data
    for dataset in config.train.datasetsTrain:
        train_loader = getTrainingTestingData('rgb', 'train', dataset.images, dataset.labels, batch_size=batch_size)
    for dataset in config.train.datasetsVal:
        test_loader = getTrainingTestingData('rgb', 'eval', dataset.images, dataset.labels, batch_size=batch_size)

    # Create a tensorboard object and Write config to tensorboard
    writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

    # Loss
    l1_criterion = nn.L1Loss()

    total_iter_num = 0
    # Start training...
    for epoch in range(config.train.numEpochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Log the current Epoch Number
        writer.add_scalar('data/Epoch Number', epoch, total_iter_num)

        # Switch to train mode
        model.train()

        end = time.time()

        running_loss = 0.0
        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()
            total_iter_num += 1

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm( depth )

            # Predict
            output = model(image)

            # Compute the loss
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            loss = (1.0 * l_ssim) + (0.1 * l_depth)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 50 == 0:
                LogProgress(model, writer, test_loader, niter)

        # Log Epoch Loss
        epoch_loss = running_loss / (len(train_loader))
        writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
        print('\nTrain Epoch Loss: {:.4f}'.format(epoch_loss))

        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

        # Save the model checkpoint every N epochs
        if (epoch % config.train.saveModelInterval) == 0:
            filename = os.path.join(CHECKPOINT_DIR, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
            if torch.cuda.device_count() > 1:
                model_params = model.module.state_dict()  # Saving nn.DataParallel model
            else:
                model_params = model.state_dict()

            torch.save(
                {
                    'model_state_dict': model_params,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'total_iter_num': total_iter_num,
                    'epoch_loss': epoch_loss,
                    'config': config_yaml
                }, filename)


def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = DepthNorm( model(image) )
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output

if __name__ == '__main__':
    main()