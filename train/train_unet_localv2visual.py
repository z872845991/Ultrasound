import os
from collections import defaultdict
import gc
import wandb
import torch
import torch.nn as nn
import numpy as np
import random
from tools.metrics import (dice_coef, get_accuracy, get_F1, get_precision,
                           get_recall, get_specificity, iou_score)
from tools.utils import AverageMeter
from torchsummary import summary
from tqdm import tqdm
import pandas as pd


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


class train_model_localv2visual():
    """Usage:
        init:model
        compile:dataloaders,criterion,optimizer,num_epochs,batch_size,train_path,val_path,device='cpu'
        fit:save_trainfile,save_testfile,save_temp_checkpoints,save_hard_seg_pic,save_checkpoints_folder
    """

    def __init__(self, model, seeds=42):
        super(train_model_localv2visual).__init__()
        self.model = model
        self.seeds = seeds
        self.name = self.model.__class__.__name__
        self.wandb = wandb
        self.wandb.init(project=self.name, entity="xiaolanshu")
        # set_seed(seeds)

    def compile(self, dataloaders, criterion, optimizer, num_epochs, batch_size, train_path, val_path, device='cpu'):
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.device = device
        self.dataloaders = dataloaders

        self.config = self.wandb.config
        self.config.batch_size = self.batch_size['train']
        self.config.test_batch_size = self.batch_size['val']
        self.config.epochs = self.num_epochs
        self.config.lr = 0.0001  # learning rate(default:0.01)
        # self.config.momentum = 0.1  # SGD momentum(default:0.5)
        # self.config.no_cuda = False  # disables CUDA training
        #self.config.seed =self.seeds
        self.config.log_interval = 10

    def summarys(self, input_size):
        summary(self.model, input_size=input_size)

    def fit(self, trainfile, testfile, tmpcheckfile, hardfile, checkfile):
        bigiou = 0
        fromnum = 0
        bigdice = 0
        fromnumd = 0
        train_columns = ['epoch', 'idx', 'loss', 'miou',
                         'maxiou', 'miniou', 'mdice', 'maxdice', 'mindice']
        val_columns = ['ACC', 'PPV', 'TNR', 'TPR', 'F1', 'miou', 'maxiou', 'miniou', 'mdice', 'maxdice', 'mindice', 'iou1', 'iou2',
                       'iou3', 'iou4', 'iou5', 'iou6', 'iou7', 'iou8', 'dice1', 'dice2', 'dice3', 'dice4', 'dice5', 'dice6', 'dice7', 'dice8']
        df = pd.DataFrame(columns=train_columns)
        val_df = pd.DataFrame(columns=val_columns)
        for epoch in range(self.num_epochs):
            print("Start epoch %d" % epoch)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    step = 0
                    epoch_loss = 0
                    avgmeter1 = AverageMeter()
                    avgmeter2 = AverageMeter()
                    for idx, data in enumerate(tqdm(self.dataloaders[phase])):
                        inputs, labels = data[0].to(
                            self.device), data[1].to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        step += 1
                        epoch_loss += loss.item()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        iou = iou_score(outputs, labels)
                        dice = dice_coef(outputs, labels)
                        avgmeter1.update(iou, self.batch_size[phase])
                        avgmeter2.update(dice, self.batch_size[phase])
                    print("loss: %5f" % (epoch_loss / step))
                    print("train: miou: %5f , midce: %5f " % (
                        avgmeter1.avg*1.0*100.0, 100.0*avgmeter2.avg*1.0))
                    # with open(trainfile, 'a+') as filetrain:
                    #     filetrain.write("epoch: %d  ,idx: %d   ,loss: %5f   ,miou:   %5f,maxiou: %5f    ,miniou: %5f    ,mdice: %5f   ,maxdice: %5f   ,mindice: %5f   " % (
                    #         epoch, idx, epoch_loss / step, avgmeter1.avg*1.0*100.0, avgmeter1.max *1.0*100.0, avgmeter1.min*100.0, avgmeter2.avg*1.0*100.0, avgmeter2.max *1.0*100.0, avgmeter2.min*100.0)+'\n')
                    value = [epoch, idx, epoch_loss/step, avgmeter1.avg*1.0, avgmeter1.max *1.0,
                             avgmeter1.min*1.0, avgmeter2.avg*1.0, avgmeter2.max *1.0, avgmeter2.min*1.0]
                    df.loc[len(df)] = value
                    df.to_csv(trainfile)
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
                    step = 0
                    with torch.no_grad():
                        for idx, data in enumerate(self.dataloaders[phase]):
                            inputs, labels = data[0].to(
                                self.device), data[1].to(self.device)
                            z = data[2]
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            step += 1
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
                    self.wandb.log({
                        "Vaild Loss": epoch_loss/step,
                        "MIOU": 100.0*te_avgmeter1.avg,
                        "MDice": 100.0*te_avgmeter2.avg
                    })
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
                    # with open(testfile, 'a+') as fileval:
                    #     fileval.write("ACC: %5f,PPV: %5f,TNR: %5f,TPR: %5f,F1: %5f,miou: %5f,maxiou: %5f,miniou: %5f,mdice: %5f,maxdice: %5f,mindice: %5f,iou1: %5f,iou2: %5f,iou3: %5f,iou4: %5f,iou5: %5f,iou6: %5f,iou7: %5f,iou8: %5f,dice1: %5f,dice2: %5f,dice3: %5f,dice4: %5f,dice5: %5f,dice6: %5f,dice7: %5f,dice8: %5f" % (
                    #         te_avgmeter3.avg*1.0*100.0, te_avgmeter4.avg*1.0*100.0, te_avgmeter5.avg*1.0 *
                    #         100.0, te_avgmeter6.avg*1.0*100.0, te_avgmeter7.avg*1.0*100.0, te_avgmeter1.avg*1.0*100.0,
                    #         te_avgmeter1.max *1.0*100.0, te_avgmeter1.min*100.0, te_avgmeter2.avg*1.0 *
                    #         100.0, te_avgmeter2.max *1.0*100.0, te_avgmeter2.min*100.0, te_avgmeter1.first*100.0,
                    #         te_avgmeter1.second*100.0, te_avgmeter1.third*100.0, te_avgmeter1.forth *
                    #         100.0, te_avgmeter1.fifth*100.0, te_avgmeter1.sixth*100.0,
                    #         te_avgmeter1.seventh*100.0, te_avgmeter1.eighth*100.0, te_avgmeter2.first *
                    #         100.0, te_avgmeter2.second*100.0, te_avgmeter2.third*100.0,
                    #         te_avgmeter2.forth*100.0, te_avgmeter2.fifth*100.0, te_avgmeter2.sixth*100.0, te_avgmeter2.seventh*100.0, te_avgmeter2.eighth*100.0) + '\n')
                    val_value = [te_avgmeter3.avg*1.0, te_avgmeter4.avg*1.0, te_avgmeter5.avg*1.0, te_avgmeter6.avg*1.0, te_avgmeter7.avg*1.0, te_avgmeter1.avg*1.0, 
                                te_avgmeter1.max *1.0, te_avgmeter1.min*1.0, te_avgmeter2.avg*1.0, te_avgmeter2.max *1.0, te_avgmeter2.min*1.0, te_avgmeter1.first*1.0,
                                 te_avgmeter1.second*1.0, te_avgmeter1.third*1.0, te_avgmeter1.forth*1.0, te_avgmeter1.fifth*1.0, te_avgmeter1.sixth*1.0,te_avgmeter1.seventh*1.0,
                                 te_avgmeter1.eighth*1.0, te_avgmeter2.first*1.0, te_avgmeter2.second*1.0, te_avgmeter2.third*1.0,
                                 te_avgmeter2.forth*1.0, te_avgmeter2.fifth*1.0, te_avgmeter2.sixth*1.0, te_avgmeter2.seventh*1.0, te_avgmeter2.eighth*1.0]
                    val_df.loc[len(val_df)]=val_value
                    val_df.to_csv(testfile)
        df['mdice'] = df['mdice'].map('{:.3%}'.format)
        df['miou'] = df['miou'].map('{:.3%}'.format)
        val_df['mdice'] = val_df['mdice'].map('{:.3%}'.format)
        val_df['miou'] = val_df['miou'].map('{:.3%}'.format)
        df.to_csv(trainfile)
        val_df.to_csv(testfile)
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
