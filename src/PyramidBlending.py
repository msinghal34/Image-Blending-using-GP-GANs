

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

import model

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pyramidBlending(src,dest,mask,net):
  
  height,width,_ = src.size()
  
  h = int(max(2**math.ceil(math.log(height,2)),2**math.ceil(math.log(width,2))))
  w = h
  
  src1 = src.cpu().numpy()
  dest1 = dest.cpu().numpy()
  mask1 = mask.cpu().numpy()
  
  src1 = cv2.resize(src1, (w, h), interpolation=cv2.INTER_LINEAR)
  dest1 = cv2.resize(dest1, (w, h), interpolation=cv2.INTER_LINEAR)
  mask1 = cv2.resize(mask1, (w, h), interpolation=cv2.INTER_NEAREST)
  
  
  comp = src1
  comp[mask1==1] = dest1[mask1==1]
  
  src_G =[]
  dest_G = []
  mask_G = []
  
  t = h
  while(t!=32):
    print(t)
    t=int(t//2)
    
    src_G.append(src1)
    dest_G.append(dest1)
    mask_G.append(mask1)
    
    src1 = cv2.pyrDown(src1)
    dest1 = cv2.pyrDown(dest1)
    mask1 = cv2.resize(mask1, (t, t), interpolation=cv2.INTER_NEAREST)
    
    
    
  comp_I = src_G[-1]
  comp_I[mask_G[-1]==1] = dest_G[-1][mask_G[-1]==1]
  comp_L = []
  
  for i in reversed(list(range(len(src_G)-1))):
    t=t*2
    src_l = cv2.subtract(src_G[i], cv2.pyrUp(src_G[i+1]))
    dest_l =  cv2.subtract(dest_G[i], cv2.pyrUp(dest_G[i+1]))
    
    comp_l = src_l
    comp_l[mask_G==1] = dest_l[mask_G==1]
    
    comp_L.append(comp_l)
  
  comp_I = comp_I/255
  

  xl = net(torch.tensor(comp_I,dtype=dtype,device=device).permute(2,0,1).unsqueeze(0))[0].permute(1,2,0).cpu().detach().numpy()
  xl = xl*255

  
  xh = xl
  for i in range(0,len(comp_L)):
    
    xl = cv2.pyrUp(xh)
    xh = cv2.add(xl,comp_L[i])
    
    
  xh = cv2.resize(xh, (width, height), interpolation=cv2.INTER_LINEAR)
  
  return xh,comp
  
