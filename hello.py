import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from dataset.Fetus import FetusDataset
# from model.seunet import Unet
# from model.unet import Unet
# from archs import NestedUNet
# from ince_unet import Unet
# from eca_unet import Unet
# from model.archs import NestedUNet
from model.Eca_att_unet import Att_Unet
# from model.attention_u_net import Att_Unet
# from model.dp_unet import Unet
# from model.ternausnet import UNet11,UNet16
# from model.r2unet import R2U_Net

# from model.res_unet import ResNet34Unet
# from model.aug_att_uent import AugAtt_Unet
# from model.self_att_unet import Att_Unet
# from model.channel_unet import myChannelUnet
# from model.cenet import CE_Net_
# from model.nolocal.unet_nonlocal_2D import unet_nonlocal_2D
from tools.metrics import dice_coef,iou_score,get_accuracy,get_precision,get_specificity,get_recall,get_F1
from tools.utils import AverageMeter
import datetime
from model.unet import Unet

x_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
   ])
Path='E:\\Broswer\\train_7m_Unet_change2_80.pth'
test_dataset = FetusDataset("F:\\test", mode='train',transform=x_transforms,target_transform=y_transforms)
dataloaders = DataLoader(test_dataset, batch_size=1)
with torch.no_grad():
    model=Unet(1) 

    data=next(iter(dataloaders))  #dataloaders val
    pic,label=data[0],data[1]

    model.load_state_dict(torch.load(Path,map_location=torch.device('cpu')))   
    y=model(pic).cpu()   

    plt.figure(3)
    plt.subplot(221)
    img_y=torch.squeeze(y).numpy()
    im=np.where(img_y >= 0, 1, 0)
    plt.imshow(img_y,cmap='gray')

    plt.subplot(222)
    img_label=label.squeeze().cpu().numpy()
    plt.imshow(img_label) 
    plt.subplot(223)
    plt.imshow(im,cmap='gray')
    plt.show()