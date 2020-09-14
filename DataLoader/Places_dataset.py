import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import csv
from PIL import Image
import os
import random


class Places205(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.data_folder  = os.path.join(self.root, 'data', 'vision', 'torralba', 'deeplearning', 'images256')
        self.split_folder = os.path.join(self.root, 'trainvalsplit_places205')
        assert(split=='train' or split=='val')
        split_csv_file = os.path.join(self.split_folder, split+'_places205.csv')
        
        self.low_shot = False
        self.transform = transform
        self.target_transform = target_transform
        with open(split_csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            self.img_files = []
            self.labels = []
            for row in reader:
                self.img_files.append(row[0])
                self.labels.append(int(row[1]))
    
    def convert_low_shot(self, k):

        label2img = {c:[] for c in range(205)}

        for n in range(len(self.labels)):
            label2img[self.labels[n]].append(self.img_files[n])

        self.img_files_lowshot = []
        self.labels_lowshot = []

        for c,imlist in label2img.items():
            random.shuffle(imlist)
            self.labels_lowshot += [c]*k
            self.img_files_lowshot += imlist[:k]        
        self.low_shot = True
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.low_shot:
            image_path = os.path.join(self.data_folder, self.img_files_lowshot[index])
            img = Image.open(image_path).convert('RGB')
            target = self.labels_lowshot[index]
        else:
            image_path = os.path.join(self.data_folder, self.img_files[index])
            img = Image.open(image_path).convert('RGB')
            target = self.labels[index]            

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.low_shot:
            return len(self.labels_lowshot)
        else:
            return len(self.labels)