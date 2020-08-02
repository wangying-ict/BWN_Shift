'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
sys.path.append('../../')
import argparse
import os
import random
import shutil
import time
import warnings
import json
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchnet.meter as tnt
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler
import models.mnist as mnist_models
from models.modules import q_modes
from examples import gen_key_map, accuracy, set_bn_eval
import wrapper

import ipdb
#torch.cuda.set_device(2)
model_names = sorted(name for name in mnist_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(mnist_models.__dict__[name]))
q_modes_choice = sorted(['kernel_wise', 'layer_wise'])
str_q_mode_map = {'layer_wise': q_modes.layer_wise,
                  'kernel_wise': q_modes.kernel_wise}

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg10_cifar10',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: vgg10_cifar10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--warmup-epoch', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine scheduler(default: step)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-after', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log-name', default='log', type=str)
parser.add_argument('--original-model', default='', type=str,
                    help='original model')
parser.add_argument('--bn-fusion', action='store_true', default=False,
                    help='ConvQ + BN fusion')
parser.add_argument('--quant-bias-scale', action='store_true', default=False,
                    help='Add Qcode for scale and quantize bias')
parser.add_argument('--extract-inner-data', action='store_true', default=False,
                    help='Extract inner feature map and weights')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpus', default=None, type=str,
                    help='GPUs id to use.You can specify multiple GPUs separated by ,')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--qw', default=4, type=int,
                    help='quantized weight bit')
parser.add_argument('--q-mode', choices=q_modes_choice, default='kernel_wise',
                    help='Quantization modes: ' +
                         ' | '.join(q_modes_choice) +
                         ' (default: kernel-wise)')

parser.add_argument('--debug', action='store_true', default=False,
                    help='save running scale in tensorboard')
parser.add_argument('--freeze-bn', action='store_true', default=False, help='Freeze BN')

best_acc1 = 0


def main():
    #print(args.resume)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None and args.gpus is None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # Simply call main_worker function
    main_worker(args.gpu, args)

def flag_tuning(layer):
    for i in range(out_channel):
        layer.flags[i][0].fill_(compute_state(layer.weight[i, :split_channel, :, :]))
        layer.flags[i][1].fill_(compute_state(layer.weight[i, split_channel:, :, :]))
        print(layer.flags)
def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    # if args.gen_map:
    #     args.qw = -1
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))

    else:
        print("=> creating model '{}'".format(args.arch))
    try:
        model = mnist_models.__dict__[args.arch](pretrained=args.pretrained)
    except KeyError:
        print('do not support {}'.format(args.arch))
        return

    print('model:\n=========\n{}\n=========='.format(model))

    if args.gpu is not None and args.gpus is None:
        #torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print('Use {} gpus'.format(args.gpus))
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        print(args.resume)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])  # GPU memory leak. todo

            if not args.quant_bias_scale:
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}) (acc: {})"
                      .format(args.resume, checkpoint['epoch'], best_acc1))
                print('=> save only weights in {}.pth'.format(args.arch))
                model.cpu()
                torch.save(model.state_dict(), '{}.pth'.format(args.arch))
                model.cuda(args.gpu)
                # save pth here
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # ConvQ + BN fusion
    if args.bn_fusion:
        print('BN fusion begin')
        model = wrapper.fuse_bn_recursively(model)
        print('after bn fusion: ')
        print(model)

    if args.resume_after:
        if os.path.isfile(args.resume_after):
            print('=> loading checkpoint {}'.format(args.resume_after))
            checkpoint = torch.load(args.resume_after, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda(args.gpu)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.extract_inner_data:
        print('extract inner feature map and weight')
        wrapper.save_inner_hooks(model)
        for k, v in model.state_dict().items():
            np.save('{}'.format(k), v.cpu().numpy())
    cudnn.benchmark = True

    # Data loading code
    print('==> Preparing data..')
#    transform_train = transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#    ])

#    transform_test = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#    ])
    '''
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers)
    args.batch_num = len(train_loader)
    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)
    '''
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/dataset/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    args.batch_num = len(train_loader)
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/dataset/mnist', train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=False)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler_step = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 300])
    scheduler_next = scheduler_step
    if args.cosine:
        scheduler_next = scheduler_cosine
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10,
                                              after_scheduler=scheduler_next)
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    if 'q' in args.arch:
        args.log_name = 'logger/{}_{}'.format(args.arch,
                                              args.log_name)
    else:
        args.log_name = 'logger/{}_{}'.format(args.arch,
                                              args.log_name)
    writer = SummaryWriter(args.log_name)
    with open('{}/{}.txt'.format(args.log_name, args.arch), 'w') as wf:
        wf.write(str(model))
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        scheduler_warmup.step()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        writer.add_scalar('val/acc1', acc1, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        if args.debug:
            cnt = 0
            for k, v in model.state_dict().items():
                if 'pos' in k or 'neg' in k or 'shift' in k:
                    writer.add_histogram(k, v, epoch)
                    cnt += 1
                    if cnt == 10:
                        break
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, prefix='{}/{}_'.format(args.log_name, args.arch))


def save_checkpoint(state, is_best, prefix, filename='checkpoint.pth.tar'):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'best.pth.tar')


def validate(val_loader, model, criterion, args):
    top1 = tnt.AverageValueMeter()
    losses = tnt.AverageValueMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.add(loss.item(), input.size(0))
            top1.add(acc1[0].item() * input.size(0), input.size(0))

            if args.extract_inner_data:
                print('early stop evaluation')
                break
            if i % args.print_freq == 0:
                print('[{}/{}] Loss: {:.4f} Acc: {:.2f}'.format(
                    i, len(val_loader), losses.mean, top1.mean))
    print('acc1: {:.4f}'.format(top1.mean))
    return top1.mean


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Training
def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()
    losses = tnt.AverageValueMeter()
    top1 = tnt.AverageValueMeter()
    # switch to train mode
    model.train()
    if args.freeze_bn:
        model.apply(set_bn_eval)
    end = time.time()
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.add(time.time() - end)
        inputs = data[0]
        #print('input size is',inputs.size())
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1 = accuracy(outputs, targets)
        losses.add(loss.item(), inputs.size(0))
        top1.add(acc1[0].item(), inputs.size(0))

        writer.add_scalar('train/loss', losses.val, base_step + i)
        writer.add_scalar('train/acc1', top1.val, base_step + i)
        if args.debug:
            for k, v in model.state_dict().items():
                if 'running_scale' in k:
                    if v.shape[0] == 1:
                        writer.add_scalar('train/{}'.format(k), v.item(), base_step + i)
                    else:
                        writer.add_histogram('train/{}'.format(k), v, base_step + i)
        # compute gradient and do SGD step
        # running_scale_list = []
        # if base_step == 0:
        #     for param in optimizer.param_groups[1]['params']:
        #         if param.grad is not None:
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        # Warning: 1. backward 2. step 3. zero_grad()

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{}] [{}/{}] Loss: {:.4f} Acc: {:.2f}'.format(
                epoch, i, args.batch_num, losses.val, top1.val))


if __name__ == '__main__':
    main()
