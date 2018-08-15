import time
import os
import copy
import argparse
import pdb
import collections
import sys

from torch.autograd import Variable
from torch.utils import data as data_
import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from data.dataset import Dataset, TestDataset, inverse_normalize
from config import opt

dataset=Dataset(opt)
dataloader = data_.DataLoader(dataset, \
                                  batch_size=opt.batch_size, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)

testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset,
                                   batch_size=opt.batch_size,
                                   num_workers=opt.test_num_workers,
                                   shuffle=False, \
                                   pin_memory=True
                                   )

resnet = model.resnet50(20,True)
resnet = resnet.cuda()
resnet = torch.nn.DataParallel(resnet).cuda()

optimizer = optim.Adam(resnet.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

loss_hist = collections.deque(maxlen=500)

resnet.train()
resnet.module.freeze_bn()

#resnet.freeze_bn()
resnet.training = True

epoch_loss = 0.0
for epoch in range(opt.epoch):
    start = time.time()
    running_loss = 0.0
    for iter_num, data in enumerate(dataloader):
        optimizer.zero_grad()
        losses = resnet([data[0].cuda().float(),data[1].cuda().float(),data[2].cuda().float(),data[3].cuda().float()])
        losses[4].backward()
        optimizer.step()

        epoch_loss += losses[4].item()
        running_loss += losses[4].item()
        if iter_num % 500 == 499:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, iter_num + 1, running_loss / 500))
            running_loss = 0.0


    print(time.time()-start)
    print('[%d] loss %.3f:'%(epoch+1, epoch_loss/5000))
    epoch_loss = 0.0
    torch.save(resnet, 'filename_'+str(epoch)+'.pt')