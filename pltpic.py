import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from dataset.Fetus import FetusDataset
# from model.unet import Unet
from model.unet_embedd_nonlocal import Unet_embedd_nonlocal
x_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
   ])
Path='E:\\Broswer\\train_7m_Unet_embedd_nonlocal_change_80.pth'
test_dataset = FetusDataset("F:\\test", mode='train',transform=x_transforms,target_transform=y_transforms)
dataloaders = DataLoader(test_dataset, batch_size=1)
with torch.no_grad():
    model=Unet_embedd_nonlocal(1) 

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