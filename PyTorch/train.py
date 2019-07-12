import time
import argparse
import datetime
import glob
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from termcolor import colored

from model import Model
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()

    # Create a new directory to save logs
    runs = sorted(glob.glob(os.path.join('logs', 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    MODEL_LOG_DIR = os.path.join('logs', 'exp-{:03d}'.format(prev_run_id + 1))
    CHECKPOINT_DIR = os.path.join(MODEL_LOG_DIR, 'checkpoints')
    os.makedirs(CHECKPOINT_DIR)
    print('Saving logs to folder: ' + colored('"{}"'.format(MODEL_LOG_DIR), 'blue'))

    # Create model
    model = Model()
    print('Model created.')

    # Enable Multi-GPU training
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        print('Multiple GPUs being used, can\'t save model graph to Tensorboard')
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    # writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)
    writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

    # Loss
    l1_criterion = nn.L1Loss()

    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm(depth)

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

            if i % 300 == 0:
                LogProgress(model, writer, test_loader, niter)

        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

        # Save the model checkpoint every N epochs
        saveModelInterval = 2
        if (epoch % saveModelInterval) == 0:
            filename = os.path.join(CHECKPOINT_DIR, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
            if torch.cuda.device_count() > 1:
                model_params = model.module.state_dict()  # Saving nn.DataParallel model
            else:
                model_params = model.state_dict()

            torch.save(
                {
                    'model_state_dict': model_params,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
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
    model.train()

if __name__ == '__main__':
    main()
