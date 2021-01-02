import os
import numpy as np 
import argparse
import shutil
import yaml
import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from data.get_dataLoader import get_dataLoader
from models.model import Resnet152
from utils import ProgressMeter, AverageMeter, accuracy



def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    filepath = os.path.join(root, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(root,'model_best.pth.tar'))



def validate(val_loader, model, criterion, print_freq, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
            
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        
    return top1.avg



def train(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)
        



def main():
    cfg = yaml.full_load(open("config.yml", 'r'))  # Load config data
    # Set logs directory
    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    dataDir = cfg['PATHS']['PROCESSED_DATA']
    experiment_Dir = os.path.join(cfg['PATHS']['RESULT'],cur_date)
    if not os.path.exists(experiment_Dir):
        os.mkdir(experiment_Dir) 


    batch_size = cfg['TRAIN']['BATCH_SIZE']
    emb_dim = cfg['NN']['RESNET152']['EMB_DIM']
    num_classes = cfg['DATA']['NUM_CLASSES']
    epochs = cfg['TRAIN']['EPOCHS']
    lr = cfg['NN']['RESNET152']['LR']
    optimizer = cfg['NN']['RESNET152']['OPTIMIZER']
    arch = cfg['TRAIN']['MODEL_DEF']
    print_freq = cfg['TRAIN']['PRINT_PREQ']
    os.environ["CUDA_VISIBLE_DEVICES"]= cfg['TRAIN']['GPU']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume = cfg['TRAIN']['CONTINUE_MODEL']
    start_epoch = 0
    best_acc1 = 0

    train_loader, val_loader = get_dataLoader(dataDir, batch_size=batch_size, workers=4)
    print(len(train_loader), len(val_loader))


    model = Resnet152(emb_dim, num_classes).to(device)  
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr,
    #                             momentum=0.9,
    #                             weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr)
                                
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    model.train()    
    for epoch in range(start_epoch, epochs):
        train(train_loader, model, criterion, optimizer, epoch, device, print_freq)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, print_freq, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        checkpoint_fname = 'ep_' + str(epoch)+ '.pth.tar'
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, experiment_Dir, filename=checkpoint_fname)
        
        


if __name__ == '__main__':
    main()
