from torch.utils.data import Dataset
from torchvision.transforms import transforms
import PIL.Image as Image
import os
import re
import cv2
import numpy as np
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import albumentations as A
def make_dataset(path):
    files = os.listdir(path)
    imgs = []
    for file in files:
        if not 'Annotation' in file:
            label = file.split('.')[0]+'_Annotation.png'
            imgs.append((file,label))

    return imgs

class Fetus_transformDataset(Dataset):
    def __init__(self,path,mode='train'):
        super(Fetus_transformDataset,self).__init__()
        self.IMAGE_RESIZE = (512, 512)
        self.RESNET_MEAN = (0.485, 0.456, 0.406)
        self.RESNET_STD = (0.229, 0.224, 0.225)
        self.path = path
        self.mode = mode
        self.imgs = make_dataset(path)
        #self.transform = Compose([Resize( self.IMAGE_RESIZE[0],  self.IMAGE_RESIZE[1])])
        self.train_transform = Compose([Resize( self.IMAGE_RESIZE[0],  self.IMAGE_RESIZE[1]),
                                   Normalize(mean=0., std=1., p=1),
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5)])
        self.test_transform=A.Compose([
            A.Resize(width=self.IMAGE_RESIZE[0],height=self.IMAGE_RESIZE[1]),
            Normalize(mean=0., std=1., p=1),
        ],p=1.0)
    def __getitem__(self, index):
        if self.mode=='train':
            img,label = self.imgs[index]
            name = img
            img=cv2.imread(os.path.join(self.path,img))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            label=cv2.imread(os.path.join(self.path,label),cv2.IMREAD_GRAYSCALE)
            label = (label >= 1).astype('float32')
            # img = Image.open(os.path.join(self.path, img)).convert("RGB")
            # # img = Image.open(os.path.join(self.path,img)).crop((200,200,1000,712))
            # # label = Image.open(os.path.join(self.path,label)).convert('L').crop((200,200,1000,712))
            # label = Image.open(os.path.join(self.path, label)).convert('L')
            if self.train_transform:
                data=self.train_transform(image=img,mask=label)
                img=data['image']
                label=data['mask']
            return np.moveaxis(np.array(img),2,0), label.reshape((1, self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]))
        else:
            img,label = self.imgs[index]
            name = img
            img=cv2.imread(os.path.join(self.path,img))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            label=cv2.imread(os.path.join(self.path,label),cv2.IMREAD_GRAYSCALE)
            label = (label >= 1).astype('float32')
            # img = Image.open(os.path.join(self.path, img)).convert("RGB")
            # # img = Image.open(os.path.join(self.path,img)).crop((200,200,1000,712))
            # # label = Image.open(os.path.join(self.path,label)).convert('L').crop((200,200,1000,712))
            # label = Image.open(os.path.join(self.path, label)).convert('L')
            if self.test_transform:
                data2=self.test_transform(image=img,mask=label)
                img=data2['image']
                label=data2['mask']
            return np.moveaxis(np.array(img),2,0), label.reshape((1, self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]))
    def __len__(self):
        return len(self.imgs)