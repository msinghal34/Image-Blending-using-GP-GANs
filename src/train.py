


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



def readBatch(batch_size, data):
  """ 
  It returns two tensors 
  First one consisting of composited images and Second original image
  Size of each tensor = batch_size X 3 X 64 X 64
  """
  folders = random.choices(list(range(len(data))), k=batch_size)
  composite = []
  original = []
  for folder_idx in folders:
    a, b = random.sample(list(range(len(data[folder_idx]))), 2)
    image_a = data[folder_idx][a]
    image_b = data[folder_idx][b]
    height , width = image_a.shape[:2]
    a, b = 16,16#random.sample(list(range(32)), 2)
    composited_image = np.array(image_a)
    composited_image[:,a:a+32, b:b+32] = image_b[:,a:a+32, b:b+32]
    composite.append(composited_image)
    original.append(image_a)
  composited = torch.tensor(composite, dtype=dtype, device=device)
  original_data = torch.tensor(original, dtype=dtype, device=device)
  return composited,original_data


def train():
	net = model.Net()
	weights_file = 'networks/network_7'
	# net.load_state_dict(torch.load('networks/network_7'))
	net.to(dtype=dtype, device=device)
	data = torch.load('../data/train.bin')
	val_data = torch.load('../data/validation.bin')

	epochs = 5
	batch_size = 16
	lr = 1e-5
	alpha, beta = 0.9,0.99
	regulirizer = 0.00001 
	optimizer = optim.SGD(net.parameters(), lr=lr)#, betas = (alpha,beta))
	num_batches = 512


	f= open(weights_file+".txt","w+")
	f.write("lr : "+str(lr))
	f.write("\nbatch_size : "+str(batch_size))
	f.write("\nalpha : "+str(alpha))
	f.write("\nbeta : "+str(beta))
	f.write("\nnum_batches : "+str(num_batches))
	# training loop
	for epoch in range(epochs):
	  t=time.time()
	  for i in range(num_batches):
	    input, target = readBatch(batch_size,data)
	    optimizer.zero_grad()
	    output = net(input)
	    criterion = nn.MSELoss()
	    l2_reg = 0
	    for param in net.parameters():
	      l2_reg += torch.norm(param)
	    loss = criterion(output, target) +  regulirizer * l2_reg
	    if (i+1)%num_batches==0:
	      t = time.time()-t
	      print(epoch,loss)
	      f.write("\nepoch "+str(epoch))
	      f.write("\n"+str(loss))
	      f.write("\ntime elapsed:"+str(t))
	      print("time elapsed:",t)
	      t = time.time()
	    loss.backward()     # Does Backpropogation
	    optimizer.step()    # Does the update
	  if((epoch+1)%1==0):
	      for pp in optimizer.param_groups:
	        pp['lr'] /= 4
	  torch.save(net.state_dict(), weights_file)
	    
	print(list((net.parameters()))[-2])
	f.close()
	torch.save(net.state_dict(), weights_file)