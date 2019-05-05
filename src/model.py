
import random
import cv2
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import matplotlib.pyplot as plt

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # Encoder
        self.batch_norm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv5 = nn.Conv2d(512, 4000, 4)
        
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        # Staring Decoder
        self.dconv5 = nn.ConvTranspose2d(4000, 512, 4)
        self.dconv4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.xavier_normal_(self.conv4.weight)
        torch.nn.init.xavier_normal_(self.conv5.weight)
        
        torch.nn.init.xavier_normal_(self.dconv1.weight)
        torch.nn.init.xavier_normal_(self.dconv2.weight)
        torch.nn.init.xavier_normal_(self.dconv3.weight)
        torch.nn.init.xavier_normal_(self.dconv4.weight)
        torch.nn.init.xavier_normal_(self.dconv5.weight)
        
        

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2)) 
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = F.relu(self.dconv5(x5))
        x7 = F.relu(self.dconv4(x6))
        x8 = F.relu(self.dconv3(x7))
        x9 = F.relu(self.dconv2(x8))
        x10 = F.relu(self.dconv1(x9))
        return x10

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
