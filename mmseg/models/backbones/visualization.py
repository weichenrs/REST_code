import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid
import cv2
from torchvision import transforms
from math import sqrt
import torch


def select(input,mode):
     if mode=='global_mean':
         output=torch.mean(input,dim=0)
         return output.unsqueeze(0)
     if mode=='var':
         var=[]
         for i in range(len(input)):
            var.append(torch.var(input[i]))
         j=var.index(max(var))
         k=var.index(min(var))
         outmax=input[j]
         outmin=input[k]
         return outmax.unsqueeze(0),outmin.unsqueeze(0)
     if mode=='mean':
         mean = []
         for i in range(len(input)):
             mean.append(torch.mean(input[i]))
         j = mean.index(max(mean))
         k = mean.index(min(mean))
         outmax = input[j]
         outmin = input[k]
         return outmax.unsqueeze(0), outmin.unsqueeze(0)
     if mode=='extreme':
         maxs = []
         mins=[]
         for i in range(len(input)):
             maxs.append(torch.max(input[i]))
             mins.append(torch.min(input[i]))
         j = maxs.index(max(maxs))
         k = mins.index(min(mins))
         outmax = input[j]
         outmin = input[k]
         return outmax.unsqueeze(0), outmin.unsqueeze(0)

def showmypic(input,orig):
    globalmean=select(input,mode='global_mean').permute(1,2,0).detach().numpy()
    varmax,varmin=select(input,mode='var')
    varmax=varmax.permute(1,2,0).detach().numpy()
    varmin=varmin.permute(1,2,0).detach().numpy()
    meanmax,meanmin=select(input,mode='mean')
    meanmax=meanmax.permute(1,2,0).detach().numpy()
    meanmin=meanmin.permute(1,2,0).detach().numpy()
    exmax,exmin=select(input,mode='extreme')
    exmax=exmax.permute(1,2,0).detach().numpy()
    exmin=exmin.permute(1,2,0).detach().numpy()
    orig=orig.permute(1,2,0)
    orig = orig.detach().numpy()
    plt.figure(figsize=(12, 12))
    g=[varmax,varmin,meanmax,meanmin,exmax,exmin,globalmean]
    g_title = ['varmax', 'varmin', 'meanmax', 'meanmin', 'exmax', 'exmin', 'globalmean']
    for i in range(len(g)):
        plt.subplot(4, 2, i+1)
        im=plt.imshow(g[i], cmap='jet')
        plt.title(g_title[i])
        plt.colorbar(im)
        plt.axis('off')
    plt.subplot(4, 2, 8)
    plt.imshow(orig)
    plt.axis('off')
    plt.show()
