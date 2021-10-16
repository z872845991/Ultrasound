from torch.utils.data import Dataset
import PIL.Image as Image
import os
import re
import cv2
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def make_dataset(path1,path2):
    imgs=[]
    label=[]
    for file in os.listdir(path1):
        img=os.path.join(path1,file)
        imgs.append(img)
    for file2 in os.listdir(path2):
        mask=os.path.join(path2,file2)
        label.append(mask)
    return imgs,label

class SDataset(Dataset):
    def __init__(self,path,mode='train',transform=None, target_transform=None):
        super(SDataset,self).__init__()
        self.imgpath=os.path.join(path,'images')
        self.maskpath=os.path.join(path,'masks')
        self.mode = mode
        self.imgs,self.label=make_dataset(self.imgpath,self.maskpath)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode=='train':
            img = self.imgs[index]
            label=self.label[index]
            name = img
            img = Image.open(img).convert("RGB")
            # img = Image.open(os.path.join(self.path,img)).crop((200,200,1000,712))
            # label = Image.open(os.path.join(self.path,label)).convert('L').crop((200,200,1000,712))
            label = Image.open(label).convert('L')
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)


            return img,label

        else:
            img, label = self.imgs[index]
            name = img
            img = Image.open(os.path.join(self.path, img))
            if self.transform is not None:
                img = self.transform(img)
            return img,name

    def __len__(self):
        return len(self.imgs)


if __name__=='__main__':
    path='F:\\sessile-main-Kvasir-SEG'
    x_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    # transforms.CenterCrop(512),
    transforms.ToTensor()
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
    # mask只需要转换为tensor
    y_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.CenterCrop(512),
        transforms.ToTensor()
    ])


    sdata=SDataset(path,transform=x_transforms,target_transform=y_transforms)
    dataset_loader = DataLoader(sdata,
                             batch_size=1,
                             shuffle=True,
                             num_workers=1)
    print(len(sdata))
    # a=next(iter(dataset_loader))
    # img,label=a[0],a[1]
    # image=img[0].numpy()
    # image=image.transpose(1,2,0)
    # print(image.shape)
    # plt.imshow()
    # plt.show()



    

