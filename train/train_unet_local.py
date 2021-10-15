import torch
import torch.nn as nn
from tools.metrics import dice_coef,iou_score,get_accuracy,get_precision,get_specificity,get_recall,get_F1
from tools.utils import AverageMeter
from torchsummary import summary
class train_model_local():
    """Usage:
        init:model
        compile:dataloaders,criterion,optimizer,num_epochs,batch_size,train_path,val_path,device='cpu'
    """
    def __init__(self,model):
        super(train_model_local).__init__()
        self.model=model
    def compile(self,dataloaders,criterion,optimizer,num_epochs,batch_size,train_path,val_path,device='cpu'):
        self.criterion=criterion
        self.optimizer=optimizer
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.train_path=train_path
        self.val_path=val_path
        self.device=device
        self.dataloaders=dataloaders
    def summarys(self,input_size):
        summary(self.model,input_size=input_size)
    def fit(self,file1,file2):
        for epoch in range(self.num_epochs):
            for phase in ['train','val']:

                if phase =='train':
                    self.model.train()
                    step=0
                    epoch_loss=0
                else:
                    self.model.eval()
                avgmeter1 = AverageMeter()
                avgmeter2 = AverageMeter()
                for idx,data in enumerate(self.dataloaders[phase]):
                    inputs,labels=data[0].to(self.device),data[1].to(self.device)
                    outputs=self.model(inputs)
                    loss=self.criterion(outputs,labels)
                    with torch.set_grad_enabled(phase=='train'):
                        step+=1
                        epoch_loss+=loss.item()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    iou=iou_score(outputs,labels)
                    dice=dice_coef(outputs,labels)
                    avgmeter1.update(iou, self.batch_size[phase])
                    avgmeter2.update(dice, self.batch_size[phase])
                if phase=='train':
                    with open(file1,'a+') as filetrain:
                        filetrain.write("epoch: %d  ,idx: %d   ,loss: %0.3f   ,miou:   %.3f,maxiou: %.3f    ,miniou: %.3f    ,mdice: %.3f   ,maxdice: %.3f   ,mindice: %.3f   " %(epoch,idx,epoch_loss / step, avgmeter1.avg, avgmeter1.max, avgmeter1.min, avgmeter2.avg, avgmeter2.max,avgmeter2.min)+'\n')
                else:
                    with open(file2,'a+') as fileval:
                        fileval.write("epoch: %d  ,idx: %d   ,miou:   %.3f,maxiou: %.3f    ,miniou: %.3f    ,mdice: %.3f   ,maxdice: %.3f   ,mindice: %.3f   " %(epoch,idx, avgmeter1.avg, avgmeter1.max, avgmeter1.min, avgmeter2.avg, avgmeter2.max,avgmeter2.min)+'\n')  