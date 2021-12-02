# import sys
# sys.path.append("D:\\OneDrive\\Github\\Ultrasound\\dataset")
import os 
import torch
from torch import nn,optim
from dataset.Fetus import FetusDataset
from torch.utils.data import DataLoader, dataloader
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from model.Teawater.Teawater_v6 import Teawater_v6
import argparse
def predict(args):
    model=Teawater_v6(1,2)
    if args.device=='cpu':
        model.load_state_dict(torch.load(args.pth,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.pth))
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
    if args.dataset=='FetusDataset':
        pre_dataset=FetusDataset(args.datapath,transform=x_transforms,target_transform=y_transforms)
    
    dataloader=DataLoader(pre_dataset,batch_size=args.batch_size,shuffle=args.shuffle)
    imgs,labels,_=next(iter(dataloader))
    imgs_y=model(imgs)
    imgs_y=nn.ReLU6()(imgs_y)
    predict_image(imgs,labels,imgs_y)

def predict_image(imgs,labels,imgs_y):
    l=len(imgs)
    for i in range(l):
        img,label,img_y=imgs[i],labels[i],imgs_y[i]
        plt.figure(i)
        plt.subplot(131)
        plt.title('Image')
        img_=img.squeeze().permute(1,2,0).numpy()
        plt.imshow(img_)
        plt.subplot(132)
        plt.title('Label')
        label_=label.squeeze().numpy()
        plt.imshow(label_)
        plt.subplot(133)
        plt.title('Predict')
        img_ys=img_y.squeeze().detach().numpy()
        plt.imshow(img_ys)
    plt.show()

if __name__=='__main__':
    parse=argparse.ArgumentParser()

    parse.add_argument('--pth',type=str,default='F:/checkpoints/train_7m_Teawater_v6_5e5_change_67.pth')
    parse.add_argument('--device',type=str,default='cpu')
    parse.add_argument('--dataset',type=str,default='FetusDataset')
    parse.add_argument('--batch_size',type=int,default=5)
    parse.add_argument('--shuffle',type=str,default='False')
    parse.add_argument('--datapath',type=str,default='E:\\Idm_Downloads\\Compressed\\Data\\data\\test')
    args=parse.parse_args()
    predict(args)