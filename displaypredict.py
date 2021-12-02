# import sys
# sys.path.append("D:\\OneDrive\\Github\\Ultrasound\\dataset")
import os 
import torch
from torch import nn,optim
from dataset.Fetus import FetusDataset
from dataset.Fetus_transform import Fetus_transformDataset
from torch.utils.data import DataLoader, dataloader
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from model.Teawater.Teawater_v6 import Teawater_v6
from matplotlib import image
import argparse
### import model
from model.unet_all_idea3_se import Unet_all_idea3_se
def predict(args,model):
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
    elif args.dataset=='Fetus_transformDataset':
        pre_dataset=Fetus_transformDataset(args.datapath,mode='val')
    dataloader=DataLoader(pre_dataset,batch_size=args.batch_size,shuffle=args.shuffle)
    imgs,labels,names=next(iter(dataloader))
    imgs_y=model(imgs)
    imgs1_y=nn.ReLU6()(imgs_y)
    imgs2_y=nn.Sigmoid()(imgs_y)
    predict_image(imgs,labels,imgs1_y,imgs2_y,names,args.flag)

def predict_image(imgs,labels,imgs1_y,imgs2_y,names,flag):
    l=len(imgs)
    for i in range(l):
        img,label,img1_y,img2_y,name=imgs[i],labels[i],imgs1_y[i],imgs2_y[i],names[i].split('.')[0]
        img_=img.squeeze().permute(1,2,0).numpy()
        label_=label.squeeze().numpy()
        img1_ys=img1_y.squeeze().detach().numpy()
        img2_ys=img2_y.squeeze().detach().numpy()
        
        plt.figure(i)
        
        plt.subplot(221)
        plt.title('Image')
        plt.imshow(img_)
        
        plt.subplot(222)
        plt.title('Label')
        plt.imshow(label_)
        
        plt.subplot(223)
        plt.title('Predict_ReLU6')
        plt.imshow(img1_ys)

        plt.subplot(224)
        plt.title('Predict_Sigmoid')
        plt.imshow(img2_ys)
        
        # path1='F:/Predict_image/%s_img_'%name+flag+'_.png'
        # path2='F:/Predict_image/%s_label_'%name+flag+'_.png'
        path3='F:/Predict_image/%s_predict_'%name+flag+'_.png'
        print(path3)
        # image.imsave(path1,img_)
        # image.imsave(path2,label_)
        image.imsave(path3,img2_ys)
        # cv2.imwrite(path1,img_)
        # cv2.imwrite(path2,label_)
        # cv2.imwrite(path3,img_ys)
    plt.show()

if __name__=='__main__':
    parse=argparse.ArgumentParser()

    parse.add_argument('--pth',type=str,default='F:/checkpoints/train_7m_Unet_all_idea3_se_1e4__change_38.pth')
    parse.add_argument('--device',type=str,default='cpu')
    parse.add_argument('--dataset',type=str,default='FetusDataset')
    parse.add_argument('--batch_size',type=int,default=5)
    parse.add_argument('--shuffle',type=str,default='True')
    parse.add_argument('--flag',type=str,default='train_7m_Unet_all_idea3_se_1e4__change_38')
    parse.add_argument('--datapath',type=str,default='E:\\Idm_Downloads\\Compressed\\Data\\data\\test')
    args=parse.parse_args()
    model=Unet_all_idea3_se(1,2)
    predict(args,model)