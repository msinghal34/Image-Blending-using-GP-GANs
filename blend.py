import sys
import os
sys.path.insert(0, './src')

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
import argparse

import PyramidBlending
import blendingSGD

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-src', help='Path to src img',dest ="src",default='')
	parser.add_argument('-dest', help='Path to src img',dest ="dest",default='')
	parser.add_argument('-mask', help='Path to mask img',dest ="mask",default='')
	parser.add_argument('-model', help='Path to model',dest ="model",default='./src/networks/network_weights_skip_1')

	args = parser.parse_args()

	net = model.Net()
	net.load_state_dict(torch.load(args.model))
	net.to(dtype=dtype, device=device)
	src = torch.tensor(cv2.cvtColor(cv2.imread(args.src), cv2.COLOR_BGR2RGB),dtype=dtype,device=device)
	dest = torch.tensor(cv2.cvtColor(cv2.imread(args.dest), cv2.COLOR_BGR2RGB),dtype=dtype,device=device)
	mask = torch.tensor(cv2.cvtColor(cv2.imread(args.mask), cv2.COLOR_BGR2RGB),dtype=dtype,device=device)

	res1,comp = PyramidBlending.pyramidBlending(src,dest,mask,net)
	res2,comp = blendingSGD.blendImageSGD(src,dest,mask,net)


	plt.figure()
	plt.imshow(src.cpu()/255)
	plt.title('source')

	plt.figure()
	plt.imshow(dest.cpu()/255)
	plt.title('Destination')

	plt.figure()
	plt.imshow(comp/255)
	plt.title('Composite Img')
	plt.savefig('./images/composite.png')

	plt.figure()
	plt.imshow(res1/255)
	plt.title('Pyramid Blending')
	plt.savefig('./images/pyramidBlending.png')

	plt.figure()
	plt.imshow(res2)
	plt.title('GP SGD')
	plt.savefig('./images/SGDBlending.png')