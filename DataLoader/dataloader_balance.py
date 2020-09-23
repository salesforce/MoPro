from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
from PIL import ImageFile
import numpy as np
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


class webvision_dataset(Dataset): 
    def __init__(self, root_dir, annotation, transform, mode): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode  
   
        if self.mode=='test':
            self.val_imgs = []
            self.val_labels = {}            
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            data = json.load(open(annotation,'r'))
            self.images = data['images']
            self.labels = data['labels']
    
    def repeat(self):
        print('=> repeating dataset')
        labels = np.array(self.labels)
        uniq,freq = np.unique(labels,return_counts=True)
        inv = (1/freq)**0.5 
        p = inv/inv.sum()
        weight = 10*p/p.min()
        weight = weight.astype(int)
        weight = {u:w for u,w in zip(uniq,weight)}
    
        train_imgs=[]
        train_labels=[]
        for im,lab in zip(self.images,self.labels):
            train_imgs += [im]*weight[lab]
            train_labels += [lab]*weight[lab]
            
        self.train_imgs=[]
        self.train_labels=[]        
        index_shuf = list(range(len(train_imgs)))
        random.shuffle(index_shuf)        
        for i in index_shuf[:len(self.labels)]:
            self.train_imgs.append(train_imgs[i])
            self.train_labels.append(train_labels[i])
        print('=> done')
    
    def __getitem__(self, index):
        if self.mode=='train':
            target = self.train_labels[index]     
            image = Image.open(self.root+self.train_imgs[index]).convert('RGB')   
            image = self.transform(image)        
            return image, target
                   
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.images)
        else:
            return len(self.val_imgs)    


class webvision_dataloader():  
    def __init__(self, batch_size, annotation, num_workers, root_dir, imagenet_dir, distributed, crop_size=0.2):

        self.batch_size = batch_size
        self.annotation = annotation
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.imagenet_dir = imagenet_dir
        self.distributed = distributed
        
        
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
     
        self.transform_test = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  

    def run(self):

        train_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="train", annotation = self.annotation)   
        test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', annotation = self.annotation)   
        imagenet_val = datasets.ImageFolder(os.path.join(self.imagenet_dir, 'val'), self.transform_test)
        
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,shuffle=False)
            imagenet_sampler = torch.utils.data.distributed.DistributedSampler(imagenet_val,shuffle=False)
        else:
            self.train_sampler = None
            eval_sampler = None
            test_sampler = None
            imagenet_sampler = None
            
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True)                                              
             
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=test_sampler)                              
         
        imagenet_loader = DataLoader(
            dataset=imagenet_val,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True,
            sampler=imagenet_sampler)                    

        
        return train_loader,test_loader,imagenet_loader     

    