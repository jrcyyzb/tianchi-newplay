# 导入包
import os, sys, glob, shutil, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2

from PIL import Image
import numpy as np

from tqdm import tqdm, tqdm_notebook

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from utils import *

# 定义好字符分类模型，使用renset18的模型作为特征提取模块

class SVHN_Model1(nn.Module):

    def __init__(self):
        super(SVHN_Model1, self).__init__()
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
        self.fc6 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc5(feat)
        return c1, c2, c3, c4, c5, c6


# 步骤4：定义好训练、验证和预测模块
def train(train_loader, model, criterion, optimizer):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = torch.IntTensor(target).long()
            target = target.cuda()
        c0, c1, c2, c3, c4, c5 = model(input)
        loss = criterion(c0, target[:, 0]) + \
            criterion(c1, target[:, 1]) + \
            criterion(c2, target[:, 2]) + \
            criterion(c3, target[:, 3]) + \
            criterion(c4, target[:, 4]) + \
            criterion(c5, target[:, 5])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(loss.item())

        train_loss.append(loss.item())
    return np.mean(train_loss)

def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = torch.tensor(target, dtype=torch.long)
                target = target.cuda()

            c0, c1, c2, c3, c4, c5 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                criterion(c1, target[:, 1]) + \
                criterion(c2, target[:, 2]) + \
                criterion(c3, target[:, 3]) + \
                criterion(c4, target[:, 4]) + \
                criterion(c5, target[:, 5])

            val_loss.append(loss.item())
    return np.mean(val_loss)


