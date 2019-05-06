

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



def getGaussian(img):
  Gaussian = torch.tensor([[[[1,2,1],[2,4,2],[1,2,1]]]],device=device,dtype=dtype)
  img_b = torch.stack([torch.nn.functional.conv2d(img[:,:,0].unsqueeze(0).unsqueeze(0), Gaussian,stride=1, padding=1)[0,0]
               ,torch.nn.functional.conv2d(img[:,:,1].unsqueeze(0).unsqueeze(0), Gaussian,stride=1, padding=1)[0,0]
               ,torch.nn.functional.conv2d(img[:,:,2].unsqueeze(0).unsqueeze(0), Gaussian,stride=1, padding=1)[0,0]]).permute(1,2,0)
  return img_b
  
def getLaplacian(img):
  Laplacian = torch.tensor([[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]],device=device,dtype=dtype)
  img_blurr = img
  return torch.stack([torch.nn.functional.conv2d(img_blurr[:,:,0].unsqueeze(0).unsqueeze(0), Laplacian,stride=1, padding=1)[0,0]
               ,torch.nn.functional.conv2d(img_blurr[:,:,1].unsqueeze(0).unsqueeze(0), Laplacian,stride=1, padding=1)[0,0]
               ,torch.nn.functional.conv2d(img_blurr[:,:,2].unsqueeze(0).unsqueeze(0), Laplacian,stride=1, padding=1)[0,0]]).permute(1,2,0)



def solveGP(Lsrc, Ldest, mask, xl):
  """
  Solving the GP equation and returns approximate solution using Gradient Descent
  src, dest and mask are torch tensors
  Lsrc.size() = [3, l, b]
  Ldest.size() = [3, l, b]
  mask.size() = [l, b]
  """ 
  
  Lsrc = Lsrc.clone().detach()
  Ldest = Ldest.clone().detach()
  mask = mask.clone().detach()
  Lcomp = Lsrc
  Lcomp[mask==1] = Ldest[mask==1]
  Lcomp = Lcomp.detach().clone()
  xl = xl.clone().detach()
  xh = xl.clone().detach().requires_grad_(True)

  
  optimizer = torch.optim.SGD([xh], lr=0.1)

  c = nn.MSELoss()

  
  for i in range(10000):
    if((i+1)%1000==0):
      optimizer.param_groups[0]['lr'] /= 2.5
    loss = c(getGaussian(xh), xl) + c(Lcomp, getLaplacian(xh))     # GP-loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return xh


def blendImageSGD(src,dest,mask,net):
  
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
    t=int(t//2)
    
    src_G.append(src1)
    dest_G.append(dest1)
    mask_G.append(mask1)
    
    src1 = cv2.pyrDown(src1)
    dest1 = cv2.pyrDown(dest1)
    mask1 = cv2.resize(mask1, (t, t), interpolation=cv2.INTER_NEAREST)
    
  src_G.append(src1)
  dest_G.append(dest1)
  mask_G.append(mask1)
    
  comp_I = src_G[-2]
  comp_I[mask_G[-2]==1] = dest_G[-2][mask_G[-2]==1]
  comp_L = []
  src_L = []
  dest_L = []
  
  for i in reversed(list(range(len(src_G)-1))):
    t=t*2
    src_l = cv2.subtract(src_G[i], cv2.pyrUp(src_G[i+1]))
    dest_l =  cv2.subtract(dest_G[i], cv2.pyrUp(dest_G[i+1]))
    
    comp_l = src_l
    comp_l[mask_G==1] = dest_l[mask_G==1]
    
    src_L.append(src_l)
    dest_L.append(dest_l)
    comp_L.append(comp_l)
  
  comp_I = comp_I/255
  
  
  xl = net(torch.tensor(comp_I,dtype=dtype,device=device).permute(2,0,1).unsqueeze(0))[0].permute(1,2,0).cpu().detach().numpy()
  xl = xl*255
  
  for i in range(0,len(comp_L)-1):
    xh = solveGP(torch.tensor(src_L[i],device=device,dtype=dtype),
                 torch.tensor(dest_L[i],device=device,dtype=dtype),
                 torch.tensor(mask_G[-i-2],device=device,dtype=dtype),
                 torch.tensor(xl,device=device,dtype=dtype))
    xl = cv2.pyrUp(xh.cpu().detach().numpy())
  
  xh = solveGP(torch.tensor(src_L[-1],device=device,dtype=dtype),
               torch.tensor(dest_L[-1],device=device,dtype=dtype),
               torch.tensor(mask_G[0],device=device,dtype=dtype),
               torch.tensor(xl,device=device,dtype=dtype))
    
  xh = cv2.resize(xh.cpu().detach().numpy(), (width, height), interpolation=cv2.INTER_LINEAR)
  
  return xh,comp
