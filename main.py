import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np
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
resnet.load_state_dict(torch.load('resnet_6.pt'))


optimizer = optim.Adam(resnet.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

loss_hist = collections.deque(maxlen=500)

resnet.train()
resnet.module.freeze_bn()

#resnet.freeze_bn()
resnet.training = True




for iter_num, data in enumerate(test_dataloader):
    losses = resnet([data[0].cuda().float(),data[1].cuda().float(),data[2].cuda().float(),data[3].cuda().float()])


    curr_loss=losses[4].item()
    loss_hist.append(float(curr_loss))


    print('Epoch: {} | Iteration: {} | loss: {:1.5f} | Running loss: {:1.5f}'.format(
            '1', iter_num, float(curr_loss), np.mean(loss_hist)))

    del curr_loss


#
#
# for epoch_num in range(opt.epoch):
#     epoch_loss = []
#     start = time.time()
#
#     for iter_num, data in enumerate(dataloader):
#         optimizer.zero_grad()
#         losses = resnet([data[0].cuda().float(),data[1].cuda().float(),data[2].cuda().float(),data[3].cuda().float()])
#         losses[4].backward()
#
#         torch.nn.utils.clip_grad_norm_(resnet.parameters(), 0.1)
#
#         optimizer.step()
#
#         curr_loss=losses[4].item()
#         loss_hist.append(float(curr_loss))
#
#         epoch_loss.append(float(curr_loss))
#
#         print('Epoch: {} | Iteration: {} | loss: {:1.5f} | Running loss: {:1.5f}'.format(
#                 epoch_num, iter_num, float(curr_loss), np.mean(loss_hist)))
#
#         del curr_loss
#
#     scheduler.step(np.mean(epoch_loss))
#     if(epoch_num%3==0):
#         torch.save(resnet.state_dict(), 'resnet_{}.pt'.format(epoch_num))