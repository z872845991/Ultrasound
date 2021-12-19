import os

import torch
import torch.nn as nn
from tools.metrics import (dice_coef, get_accuracy, get_F1, get_precision,
                           get_recall, get_specificity, iou_score)
from tools.utils import AverageMeter
from torchsummary import summary


class train_model_localv2():
    """Usage:
        init:model
        compile:dataloaders,criterion,optimizer,num_epochs,batch_size,train_path,val_path,device='cpu'
        fit:save_trainfile,save_testfile,save_temp_checkpoints,save_hard_seg_pic,save_checkpoints_folder
    """

    def __init__(self, model):
        super(train_model_localv2).__init__()
        self.model = model

    def compile(self, dataloaders, criterion, optimizer, num_epochs, batch_size, train_path, val_path, device='cpu'):
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.device = device
        self.dataloaders = dataloaders

    def summarys(self, input_size):
        summary(self.model, input_size=input_size)

    def fit(self, trainfile, testfile, tmpcheckfile, hardfile, checkfile):
        bigiou = 0
        fromnum = 0
        bigdice = 0
        fromnumd = 0
        for epoch in range(self.num_epochs):
            print("Start epoch %d" % epoch)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    step = 0
                    epoch_loss = 0
                    avgmeter1 = AverageMeter()
                    avgmeter2 = AverageMeter()
                    for idx, data in enumerate(self.dataloaders[phase]):
                        inputs, labels = data[0].to(
                            self.device), data[1].to(self.device)
                        dp4, dp3, dp2, dp1 = self.model(inputs)
                        loss = self.criterion(dp4, dp3, dp2, dp1, labels)
                        step += 1
                        epoch_loss += loss.item()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        iou = iou_score(outputs, labels)
                        dice = dice_coef(outputs, labels)
                        avgmeter1.update(iou, self.batch_size[phase])
                        avgmeter2.update(dice, self.batch_size[phase])
                    print("loss: %5f,train: miou: %5f , midce: %5f " % (
                        epoch_loss / step, avgmeter1.avg*100.0, 100.0*avgmeter2.avg))
                    with open(trainfile, 'a+') as filetrain:
                        filetrain.write("epoch: %d  ,idx: %d   ,loss: %5f   ,miou:   %5f,maxiou: %5f    ,miniou: %5f    ,mdice: %5f   ,maxdice: %5f   ,mindice: %5f   " % (
                            epoch, idx, epoch_loss / step, avgmeter1.avg*100.0, avgmeter1.max*100.0, avgmeter1.min*100.0, avgmeter2.avg*100.0, avgmeter2.max*100.0, avgmeter2.min*100.0)+'\n')
                else:
                    self.model.eval()
                    threshold = 0.5
                    te_avgmeter1 = AverageMeter()
                    te_avgmeter2 = AverageMeter()
                    te_avgmeter3 = AverageMeter()
                    te_avgmeter4 = AverageMeter()
                    te_avgmeter5 = AverageMeter()
                    te_avgmeter6 = AverageMeter()
                    te_avgmeter7 = AverageMeter()
                    epoch_loss = 0
                    with torch.no_grad():
                        for idx, data in enumerate(self.dataloaders[phase]):
                            inputs, labels = data[0].to(
                                self.device), data[1].to(self.device)
                            z = data[2]
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            epoch_loss += loss.item()
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
                            if epoch == (self.num_epochs-1):
                                if iou1 < threshold:
                                    with open(hardfile, 'a+') as file:
                                        for s in z:
                                            file.write(s + '\n')

                    print("test: miou: %5f , midce: %5f " %
                          (100.0*te_avgmeter1.avg, 100.0*te_avgmeter2.avg))
                    if bigiou < (100.0*te_avgmeter1.avg):
                        bigiou = 100.0*te_avgmeter1.avg
                        fromnum = epoch
                        savepth = tmpcheckfile+'_%d.pth' % epoch
                        torch.save(self.model.state_dict(), savepth)
                    if bigdice < (100.0*te_avgmeter2.avg):
                        bigdice = 100.0*te_avgmeter2.avg
                        fromnumd = epoch
                        if fromnum != fromnumd:
                            savepth1 = tmpcheckfile+'_%d.pth' % epoch
                            torch.save(self.model.state_dict(), savepth1)
                    with open(testfile, 'a+') as fileval:
                        fileval.write("ACC: %5f,PPV: %5f,TNR: %5f,TPR: %5f,F1: %5f,miou: %5f,maxiou: %5f,miniou: %5f,mdice: %5f,maxdice: %5f,mindice: %5f,iou1: %5f,iou2: %5f,iou3: %5f,iou4: %5f,iou5: %5f,iou6: %5f,iou7: %5f,iou8: %5f,dice1: %5f,dice2: %5f,dice3: %5f,dice4: %5f,dice5: %5f,dice6: %5f,dice7: %5f,dice8: %5f" % (
                            te_avgmeter3.avg*100.0, te_avgmeter4.avg*100.0, te_avgmeter5.avg *
                            100.0, te_avgmeter6.avg*100.0, te_avgmeter7.avg*100.0, te_avgmeter1.avg*100.0,
                            te_avgmeter1.max*100.0, te_avgmeter1.min*100.0, te_avgmeter2.avg *
                            100.0, te_avgmeter2.max*100.0, te_avgmeter2.min*100.0, te_avgmeter1.first*100.0,
                            te_avgmeter1.second*100.0, te_avgmeter1.third*100.0, te_avgmeter1.forth *
                            100.0, te_avgmeter1.fifth*100.0, te_avgmeter1.sixth*100.0,
                            te_avgmeter1.seventh*100.0, te_avgmeter1.eighth*100.0, te_avgmeter2.first *
                            100.0, te_avgmeter2.second*100.0, te_avgmeter2.third*100.0,
                            te_avgmeter2.forth*100.0, te_avgmeter2.fifth*100.0, te_avgmeter2.sixth*100.0, te_avgmeter2.seventh*100.0, te_avgmeter2.eighth*100.0) + '\n')
        # 将最大的miou和mdice从临时文件夹移走
        oldname = tmpcheckfile+'_%d.pth' % fromnum
        newname = checkfile+'_%d.pth' % fromnum
        os.rename(oldname, newname)
        self.wandb.save(oldname)
        if fromnum != fromnumd:
            oldname2 = tmpcheckfile+'_%d.pth' % fromnumd
            newname2 = checkfile+'_%d.pth' % fromnumd
            os.rename(oldname2, newname2)
            self.wandb.save(newname2)

        print("The max Mean IOU is:%.4f" % bigiou)
        print("The number epoch is:%d" % fromnum)
        print("The max Mean Dice is:%.4f" % bigdice)
        print("The number epoch is:%d" % fromnumd)
        return fromnum
