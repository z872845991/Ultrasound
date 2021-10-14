import os
import torch
import torch.nn as nn
from tools.metrics import dice_coef,iou_score,get_accuracy,get_precision,get_specificity,get_recall,get_F1
from tools.utils import AverageMeter
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib
class predict():
    """Usage:
        init:model
        compile:dataloaders,criterion,optimizer,num_epochs,batch_size,train_path,val_path,device='cpu'
    """
    def __init__(self,model):
        super(predict).__init__()
        self.model=model
    def predictimage(self,dataloaders,path,topath,device="cpu"):
        self.model.load_state_dict(torch.load(path))
        self.model.to(device)
        plt.ion()
        avgmeter = AverageMeter()
        with torch.no_grad():
            for input,name in dataloaders:
                input.to(device)
                y=self.model(input)

                y=y.cpu()
                img_y=torch.squeeze(y).numpy()
                matplotlib.image.imsave(topath+'unet_%s.png'%name,img_y)
