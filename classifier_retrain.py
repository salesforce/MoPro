#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from resnet import *
import DataLoader.dataloader_balance as dataloader

import tensorboard_logger as tb_logger

import numpy as np


parser = argparse.ArgumentParser(description='PyTorch WebVision Classifier Retraining')
parser.add_argument('--data', default='../WebVision/dataset/',
                    help='path to WebVision dataset')
parser.add_argument('--imagenet', default='',
                    help='path to ImageNet validation set')

parser.add_argument('--annotation', default='./pseudo_label.json',
                    help='path to pseudo-label annotation')

parser.add_argument('--exp-dir', default='experiment/cRT', type=str,
                    help='experiment directory')

parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=15, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[5, 10], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--num-class', default=1000, type=int)
parser.add_argument('--cos', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='finetune encoder')


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass    

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
  
    # create model
    print("=> creating resnet model")
    encoder = resnet50(encoder=True)       
    classifier = nn.Linear(2048,1000)
    classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.bias.data.zero_()
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            encoder.cuda(args.gpu)
            classifier.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            state_dict = checkpoint['state_dict']    
            
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q'):
                    # remove prefix
                    state_dict[k.replace('.encoder_q','')] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]  
            for k in list(state_dict.keys()):
                if k.startswith('module.classifier'):
                    # remove prefix
                    state_dict[k.replace('.classifier','')] = state_dict[k]
                    # delete renamed k
                    del state_dict[k]       
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))            
            classifier.load_state_dict(state_dict,strict=False)      
            encoder.load_state_dict(state_dict,strict=False)                     
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    
    cudnn.benchmark = True
    
    if args.finetune:
        optimizer_encoder = torch.optim.SGD(encoder.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)        

    optimizer = torch.optim.SGD(classifier.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # Data loading code
    loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_workers=args.workers,root_dir=args.data,\
                                             imagenet_dir=args.imagenet,distributed=args.distributed,annotation=args.annotation)           
    train_loader,test_loader,imagenet_loader = loader.run()   
    
    if args.gpu==0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)
    else:
        logger = None
  
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            loader.train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
                
        train(train_loader, encoder, classifier, criterion, optimizer, epoch, args, logger)
    
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': classifier.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir,epoch))
        test(encoder, classifier, test_loader, imagenet_loader, args, epoch, logger)

def train(train_loader, encoder, classifier, criterion, optimizer, epoch, args, tb_logger):
    train_loader.dataset.repeat()
    
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    losses_cls = AverageMeter('Loss@Cls', ':2.2f')
    acc_cls = AverageMeter('Acc@Cls', ':4.2f')

    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_cls, acc_cls],
        prefix="Epoch: [{}]".format(epoch))

    if args.finetune:
        # finetune encoder backbone
        encoder.train()
    else:
        encoder.eval()
    classifier.train()
    
    end = time.time()
    for i, (img,target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = img.cuda(args.gpu, non_blocking=True) 
        target = target.cuda(args.gpu, non_blocking=True)             
        if args.finetune:
            feature = encoder(img)
        else:    
            with torch.no_grad():
                feature = encoder(img)
        output = classifier(feature)
       
        loss = criterion(output, target)    

        losses_cls.update(loss.item())
        
        acc = accuracy(output, target)[0] 
        acc_cls.update(acc[0])
        
        # compute gradient and do SGD step
        if args.finetune:
            optimizer_encoder.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.finetune:
            optimizer_encoder.step()        
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
            
    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        
    
def test(encoder, classifier, test_loader, imagenet_loader, args, epoch, tb_logger):
    with torch.no_grad():
        encoder.eval()
        classifier.eval()    
        top1_webvision = AverageMeter('Top1@webvision', ':4.2f')
        top5_webvision = AverageMeter('Top5@webvision', ':4.2f')
        top1_imagenet = AverageMeter('Top1@imagenet', ':4.2f')
        top5_imagenet = AverageMeter('Top5@imagenet', ':4.2f')
        print('==> Evaluation...')      
        
        # evaluate on webvision val set
        for batch_idx, (img,target) in enumerate(test_loader):
            img = img.cuda(args.gpu, non_blocking=True) 
            target = target.cuda(args.gpu, non_blocking=True) 
            feature = encoder(img)
            outputs = classifier(feature)
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            top1_webvision.update(acc1[0])
            top5_webvision.update(acc5[0])
        
        # evaluate on imagenet val set
        for batch_idx, (img,target) in enumerate(imagenet_loader):
            img = img.cuda(args.gpu, non_blocking=True) 
            target = target.cuda(args.gpu, non_blocking=True) 
            feature = encoder(img)
            outputs = classifier(feature)          
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            top1_imagenet.update(acc1[0])
            top5_imagenet.update(acc5[0])
              
        acc_tensors = torch.Tensor([top1_webvision.avg,top5_webvision.avg,top1_imagenet.avg,top5_imagenet.avg]).cuda(args.gpu)
        dist.all_reduce(acc_tensors)
        
        acc_tensors /= args.world_size
        
        print('Webvision Accuracy is %.2f%% (%.2f%%)'%(acc_tensors[0],acc_tensors[1]))
        print('ImageNet Accuracy is %.2f%% (%.2f%%)'%(acc_tensors[2],acc_tensors[3]))
        if args.gpu ==0:
            tb_logger.log_value('WebVision top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('WebVision top5 Acc', acc_tensors[1], epoch)    
            tb_logger.log_value('ImageNet top1 Acc', acc_tensors[2], epoch)
            tb_logger.log_value('ImageNet top5 Acc', acc_tensors[3], epoch)              
    return   


    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    
if __name__ == '__main__':
    main()
