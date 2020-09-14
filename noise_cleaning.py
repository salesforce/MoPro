#!/usr/bin/env python
import argparse
import os
import random
import tensorboard_logger as tb_logger
import json
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import MoPro
from resnet import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser(description='PyTorch WebVision Noise Correction')
parser.add_argument('--data', metavar='dir', default='../WebVision/dataset/',
                    help='path to webvision dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',choices=['resnet50',])
parser.add_argument('-j', '--workers', default=16, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('--num-class', default=1000, type=int)
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples')

parser.add_argument('--pseudo_th', default=0.8, type=float,
                    help='threshold for pseudo labels')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='weight to combine model prediction and prototype prediction')
parser.add_argument('--temperature', default=0.1, type=float,
                    help='contrastive temperature')
parser.add_argument('--annotation', default='./pseudo_label.json',
                    help='path to pseudo-label annotation')

class webvision_dataset(Dataset): 
    def __init__(self, root_dir): 
        self.root = root_dir
        self.transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
        self.train_imgs = []
        self.train_labels = {}            
        with open(self.root+'info/train_filelist_google.txt') as f:
            lines=f.readlines()    
            for line in lines:
                img, target = line.split()
                target = int(target)
                self.train_imgs.append(img)
                self.train_labels[img]=target            

        with open(self.root+'info/train_filelist_flickr.txt') as f:
            lines=f.readlines()    
            for line in lines:
                img, target = line.split()
                target = int(target)
                self.train_imgs.append(img)
                self.train_labels[img]=target            
   
    def __getitem__(self, index):
        img_path = self.train_imgs[index]
        target = self.train_labels[img_path]     
        image = Image.open(self.root+img_path).convert('RGB')   
        img = self.transform(image)        
        return img, target, img_path
    def __len__(self):
        return len(self.train_imgs)

        
def main():
    args = parser.parse_args()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50':
        model = MoPro(resnet50,args,width=1)
    elif args.arch == 'resnet50x2':    
        model = MoPro(resnet50,args,width=2)
    elif args.arch == 'resnet50x4':    
        model = MoPro(resnet50,args,width=4)
    else:
        raise NotImplementedError('model not supported {}'.format(args.arch))       
        
    model = model.cuda(args.gpu)
    model.eval()   
    
    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]        
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
  
    # Data loading code
    dataset = webvision_dataset(root_dir=args.data)
    loader = DataLoader(dataset=dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)    
    
    images = []
    labels = []
    print("=> performing noise cleaning on the training data")
    with torch.no_grad():   
        for (img, target, img_path) in tqdm(loader):
            img = img.cuda(args.gpu, non_blocking=True) 
            target = target.cuda(args.gpu, non_blocking=True) 
            output,feat = model.encoder_q(img)            
            
            logits = torch.mm(feat,model.prototypes.t())/args.temperature      
            soft_label = (F.softmax(output, dim=1)+F.softmax(logits,dim=1))/2

            gt_score = soft_label[target>=0,target]
            clean_idx = gt_score>(1/args.num_class)     
            
            max_score, hard_label = soft_label.max(1)
            correct_idx = max_score>args.pseudo_th
            target[correct_idx] = hard_label[correct_idx]
            clean_idx = clean_idx | correct_idx  
            for clean,label,path in zip(clean_idx,target.cpu(),img_path):
                if clean:
                    images.append(path)
                    labels.append(label.item())   
    
    json.dump({'images':images,'labels':labels},open(args.annotation,'w'))
    print("=> pseudo-label annotation saved to {}".format(args.annotation))
    
if __name__ == '__main__':
    main()
