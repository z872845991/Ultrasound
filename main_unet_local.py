import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torchsummary import summary
from dataset.Fetus import FetusDataset
from model.unet import Unet
from tools.metrics import dice_coef,iou_score,get_accuracy,get_precision,get_specificity,get_recall,get_F1
from tools.utils import AverageMeter
from model.idea3 import Idea3
from model.unet import Unet
from model.unet_res_myself import Unet_res_myself
from model.unet_res_myself_pool_connect_o_C import Unet_res_myself_pool
from train.train_unet_local import train_model_local
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_epoch=80
batch_size={'train':2,
            'val':1
            }
train_path='/home/p920/cf/data/train/'
val_path='/home/p920/cf/data/test/'

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

train_dataset=FetusDataset(train_path,transform=x_transforms,target_transform=y_transforms)
val_dataset=FetusDataset(val_path,transform=x_transforms,target_transform=y_transforms)
dataloaders={
    'train':DataLoader(train_dataset,batch_size=batch_size['train'],shuffle=True,num_workers=0),
    'val':DataLoader(val_dataset,batch_size=batch_size['val'],shuffle=True,num_workers=0)
             }

#model1
model=Unet(n_class=1)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

train1=train_model_local(model)
train1.compile(dataloaders,criterion,optimizer,num_epoch,batch_size,train_path,val_path,device)
train1.fit('result2/train/train_7m_unet.txt','result2/val/val_7m_unet.txt')


#mode2
model2=Unet_res_myself_pool(n_class=1)
model2.to(device)

train2=train_model_local(model2)
train2.compile(dataloaders,criterion,optimizer,num_epoch,batch_size,train_path,val_path,device)
train2.fit('result2/train/train_7m_unet_res_myself_pool.txt','result2/val/val_7m_unet_res_myself_pool.txt')

#model3
model3=Idea3(n_class=1)
model3.to(device)

train3=train_model_local(model3)
train3.compile(dataloaders,criterion,optimizer,num_epoch,batch_size,train_path,val_path,device)
train3.fit('result2/train/train_7m_Idea3.txt','result2/val/val_7m_Idea3.txt')