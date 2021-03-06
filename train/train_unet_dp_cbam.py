import os
import torch
import torch.nn as nn
from tools.metrics import dice_coef,iou_score,get_accuracy,get_precision,get_specificity,get_recall,get_F1
from tools.utils import AverageMeter
from torchsummary import summary

def muti_bce_loss_fusion(out, d1, d2, d3,  labels_v):

    bce_loss = nn.BCEWithLogitsLoss()
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)

    loss_out = bce_loss(out ,labels_v)

    loss = 0.1*loss1 + 0.2*loss2 + 0.3*loss3 + 0.4*loss_out
    # print("l1: %3f, l2: %3f, l3: %3f, lout: %3f\n"%(loss1.item(),loss2.item(),loss3.item(),loss_out.item()))

    return loss
class train_model_mutiloss():
    """Usage:
        init:model
        compile:dataloaders,criterion,optimizer,num_epochs,batch_size,train_path,val_path,device='cpu'
    """
    def __init__(self,model):
        super().__init__()
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
    def fit(self,trainfile,testfile,tmpcheckfile,hardfile,checkfile):
        bigiou=0
        fromnum=0
        for epoch in range(self.num_epochs):
            print("Start epoch %d"%epoch)
            for phase in ['train','val']:
                if phase =='train':
                    self.model.train()
                    step=0
                    epoch_loss=0
                    avgmeter1 = AverageMeter()
                    avgmeter2 = AverageMeter()
                    for idx,data in enumerate(self.dataloaders[phase]):
                        inputs,labels=data[0].to(self.device),data[1].to(self.device)
                        outputs, dp1, dp2, dp3 = self.model(inputs)
                        loss2, loss = muti_bce_loss_fusion(outputs, dp1, dp2, dp3, labels)
                        step+=1
                        epoch_loss+=loss.item()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        iou=iou_score(outputs,labels)
                        dice=dice_coef(outputs,labels)
                        avgmeter1.update(iou, self.batch_size[phase])
                        avgmeter2.update(dice, self.batch_size[phase])
                    print("train: miou: %5f , midce: %5f "%(avgmeter1.avg*100.0,100.0*avgmeter2.avg))
                    with open(trainfile,'a+') as filetrain:
                        filetrain.write("epoch: %d  ,idx: %d   ,loss: %0.3f   ,miou:   %.3f,maxiou: %.3f    ,miniou: %.3f    ,mdice: %.3f   ,maxdice: %.3f   ,mindice: %.3f   " %(epoch,idx,epoch_loss / step, avgmeter1.avg, avgmeter1.max, avgmeter1.min, avgmeter2.avg, avgmeter2.max,avgmeter2.min)+'\n')
                else:
                    self.model.eval()
                    threshold=0.5
                    te_avgmeter1 = AverageMeter()
                    te_avgmeter2 = AverageMeter()
                    te_avgmeter3 = AverageMeter()
                    te_avgmeter4 = AverageMeter()
                    te_avgmeter5 = AverageMeter()
                    te_avgmeter6 = AverageMeter()
                    te_avgmeter7 = AverageMeter()
                    epoch_loss=0
                    with torch.no_grad():
                        for idx,data in enumerate(self.dataloaders[phase]):
                            inputs,labels=data[0].to(self.device),data[1].to(self.device)
                            z=data[2]
                            outputs, dp1, dp2, dp3  =self.model(inputs)
                            loss=self.criterion(outputs,labels)
                            epoch_loss+=loss.item()
                            iou1 = iou_score(outputs, labels)
                            dice1 = dice_coef(outputs, labels)
                            ACC1 = get_accuracy(outputs, labels)
                            PPV1 = get_precision(outputs, labels)
                            TNR1 = get_specificity(outputs, labels)
                            TPR1 = get_recall(outputs, labels)
                            F11 = get_F1(outputs, labels)
                            te_avgmeter1.update(iou1)
                            te_avgmeter2.update(dice1)
                            te_avgmeter3.update(ACC1)
                            te_avgmeter4.update(PPV1)
                            te_avgmeter5.update(TNR1)
                            te_avgmeter6.update(TPR1)
                            te_avgmeter7.update(F11)
                            if epoch==(self.num_epochs-1):
                                if iou1 < threshold:
                                    with open(hardfile, 'a+') as file:
                                        for s in z:
                                            file.write(s + '\n')

                    print("test: miou: %5f , midce: %5f "%(100.0*te_avgmeter1.avg,100.0*te_avgmeter2.avg))
                    if bigiou<(100.0*te_avgmeter1.avg):
                        bigiou=100.0*te_avgmeter1.avg
                        fromnum=epoch
                        savepth=tmpcheckfile+'_%d.pth'%epoch
                        torch.save(self.model.state_dict(),savepth)
                    with open(testfile,'a+') as fileval:
                        fileval.write("ACC:%.4f,PPV:%.4f,TNR:%.4f,TPR:%.4f,F1:%.4f,miou:%.4f,maxiou:%.4f,miniou:%.4f,mdice:%.4f,maxdice:%.4f,mindice:%.4f,iou1:%.4f,iou2:%.4f,iou3:%.4f,iou4:%.4f,iou5:%.4f,iou6:%.4f,iou7:%.4f,iou8:%.4f,dice1:%.4f,dice2:%.4f,dice3:%.4f,dice4:%.4f,dice5:%.4f,dice6:%.4f,dice7:%.4f,dice8:%.4f" % (
                            te_avgmeter3.avg, te_avgmeter4.avg, te_avgmeter5.avg, te_avgmeter6.avg, te_avgmeter7.avg, te_avgmeter1.avg, te_avgmeter1.max, te_avgmeter1.min, te_avgmeter2.avg, te_avgmeter2.max,te_avgmeter2.min, te_avgmeter1.first, te_avgmeter1.second, te_avgmeter1.third, te_avgmeter1.forth, te_avgmeter1.fifth, te_avgmeter1.sixth, te_avgmeter1.seventh, te_avgmeter1.eighth, te_avgmeter2.first, te_avgmeter2.second, te_avgmeter2.third, te_avgmeter2.forth, te_avgmeter2.fifth, te_avgmeter2.sixth, te_avgmeter2.seventh, te_avgmeter2.eighth) + '\n')
        oldname=tmpcheckfile+'_%d.pth'%fromnum
        newname=checkfile+'_%d.pth'%fromnum
        os.rename(oldname,newname)
        print("The max Mean IOU is:%.4f"%bigiou)
        print("The number epoch is:%d"%fromnum)
        return fromnum
