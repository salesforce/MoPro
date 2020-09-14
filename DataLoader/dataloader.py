from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

    
class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir+'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root+str(c))
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,str(c),img)])                
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)
    

class webvision_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, num_class, transform_strong=None): 
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
                    if target<num_class:
                        self.val_imgs.append(img)
                        self.val_labels[img]=target                             
        else:    
            self.transform_strong = transform_strong
            self.train_imgs = []
            self.train_labels = {}            
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target<num_class:
                        self.train_imgs.append(img)
                        self.train_labels[img]=target            
            
            with open(self.root+'info/train_filelist_flickr.txt') as f:
                lines=f.readlines()    
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target<num_class:
                        self.train_imgs.append(img)
                        self.train_labels[img]=target            
   
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)        
            img_aug = self.transform_strong(image)    
            return img, target, img_aug 
                   
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    


class webvision_dataloader():  
    def __init__(self, batch_size, num_class, num_workers, root_dir, distributed, crop_size=0.2):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.distributed = distributed
        
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),            
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

        train_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="train", num_class=self.num_class, transform_strong = self.transform_strong)   
        test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', num_class=self.num_class) 
        imagenet_val = imagenet_dataset(root_dir=self.root_dir, transform=self.transform_test, num_class=self.num_class)  
        
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

    