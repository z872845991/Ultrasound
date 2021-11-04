import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torchsummary import summary
from dataset.Fetus import FetusDataset
from tools.metrics import dice_coef,iou_score,get_accuracy,get_precision,get_specificity,get_recall,get_F1
from tools.utils import AverageMeter
from train.train_unet_local import train_model_local
import matplotlib.pyplot as plt
import datetime
import numpy as np
'''
导入模型
'''
from model.unet_res_myself_pool_connect_o_C import Unet_res_myself_pool
from model.unet import Unet
from model.idea3 import Idea3
from model.idea3 import Idea3_seblock
from model.unet_self_attention import Unet_self_attention
from model.unet_encoder_idea3_se import Unet_encoder_idea3_se
from model.unet_decoder_se import Unet_decoder_se
''''''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#
#
num_epoch=81
batch_size={'train':2,
            'val':1
            }
train_path='/content/drive/MyDrive/Data/Data/data/train'
val_path='/content/drive/MyDrive/Data/Data/data/test'
#变换
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
#加载数据集
train_dataset=FetusDataset(train_path,transform=x_transforms,target_transform=y_transforms)
val_dataset=FetusDataset(val_path,transform=x_transforms,target_transform=y_transforms)
#设置dataloader
dataloaders={
    'train':DataLoader(train_dataset,batch_size=batch_size['train'],shuffle=True,num_workers=0),
    'val':DataLoader(val_dataset,batch_size=batch_size['val'],shuffle=True,num_workers=0)
             }
#
#
def main():
    filepre=['/content/drive/MyDrive/result/train/train_7m_','/content/drive/MyDrive/result/test/test_7m_','/content/drive/MyDrive/result/checkpoints/train_7m_','/content/drive/MyDrive/result/hard/hard_']
    filelast=['_change.csv','_change.csv','_change','_change.csv']
    A=[fun1(),fun2(),fun3()]
    for i in range(len(A)):
        start = datetime.datetime.now()
        name=A[i].__class__.__name__
        file = []
        for j in range(4):
            file.append((filepre[j]+name+filelast[j]))
        model=A[i]
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters())
        train = train_model_local(model)
        train.compile(dataloaders, criterion, optimizer, num_epoch, batch_size, train_path, val_path, device)
        end = datetime.datetime.now()
        t = (end - start).total_seconds()
        print("No. %d,%d min, %d seconds" % (i+1,(t / 60), (t % 60)))
        path = file[2] + '_%d.pth' % (num_epoch - 1)
        with torch.no_grad():
            data = next(iter(dataloaders['val']))
            pic3, label = data[0].to(device), data[1].to(device)

            model.load_state_dict(torch.load(path))
            y = model(pic3).cpu()

            plt.figure(i)
            plt.subplot(131)
            img3_y = torch.squeeze(y).numpy()
            im3 = np.where(img3_y >= 0, 1, 0)
            plt.imshow(img3_y)

            plt.subplot(132)
            img_label = label.squeeze().cpu().numpy()
            plt.imshow(img_label)
            plt.subplot(133)
            plt.imshow(im3, cmap='gray')
if __name__=='__main__':
    main()