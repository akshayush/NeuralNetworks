import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import tqdm_notebook 
import seaborn as sns
import time
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

import torch
from models.FirstNNs import FirstNetwork_Parameters,FirstNetwork_Linear_Optim,FirstNetwork_Sequential
from utils.utils import accuracy, cnn_fit,get_test_train_data,fit,fit_optim,fit_v3,CIFAR10_data
from torch import optim
import utils.interpreting_cnn
from models.FirstCNNs import LeNet
import torchvision.models 
import torch.nn as nn
from models.large_cnn import CIFAR10_data_transformed,gn_evaluation,large_cnn_fit,vgg_model,resnet18_model,inceptionv3_model


def __main__():
  utils.interpreting_cnn.cam('gradcamplusplus')

## Please uncomment the below lines of code for Le Net
## Inception_v3

  #torch.manual_seed(0)
  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #print(device)
  #net=inceptionv3_model()
  #net=net.to(device)
  #tic=time.time()
  #trainloader,testloader=CIFAR10_data_transformed(batchsize=16,size=299)
  #large_cnn_fit(trainloader,testloader,net,device,model="Inception",lr=0.01)
  #toc=time.time()
  #print("Time Taken is "+str(toc-tic))


## Please uncomment the below lines of code for Le Net
## VGG
  #torch.manual_seed(0)
  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #print(device)
  #net=vgg_model()
  #net=net.to(device)
  #tic=time.time()
  #trainloader,testloader=CIFAR10_data_transformed(batchsize=8)
  #large_cnn_fit(trainloader,testloader,net,device,model="VGG",lr=0.05)
  #toc=time.time()
  #print("Time Taken is "+str(toc-tic))

## Please uncomment the below lines of code for Le Net
## ResNet
  #torch.manual_seed(0)
  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #print(device)
  #net=resnet18_model()
  #net=net.to(device)
  #tic=time.time()
  #trainloader,testloader=CIFAR10_data_transformed(batchsize=16)
  #large_cnn_fit(trainloader,testloader,net,device,model="Resnet",lr=0.01)
  #toc=time.time()
  #print("Time Taken is "+str(toc-tic))

## Please uncomment the below lines of code for Le Net
##LeNet
  #torch.manual_seed(0)
  #net=LeNet()
  #tic=time.time()
  #trainloader,testloader=CIFAR10_data(batchsize=128)
  #cnn_fit(trainloader,testloader,net)
  #toc=time.time()
  #print("Time Taken is "+str(toc-tic))
#LeNet

  #fn = FirstNetwork_Parameters() ## FirstNetwork_Linear_Optim
  #fn = FirstNetwork_Linear_Optim() ## 
  #fn=FirstNetwork_Sequential()
  #(X_train, X_val, Y_train, Y_val) = get_test_train_data(random_state=0)
  #device=torch.device("cuda")
  #print("Running on "+ str(device))
  #X_train=X_train.to(device)
  #Y_train=Y_train.to(device)
  #fn.to(device)
  #fit(X_train,Y_train,fn)
  
  #loss_fn=F.cross_entropy
  #opt=optim.SGD(fn.parameters(),lr=1)
  #loss=fit_v3(X_train,Y_train,fn,opt,loss_fn)
  #toc=time.time()
  #print("Final Loss "+str(loss))
  #print("Time Taken is "+str(toc-tic))
  
  
  
__main__()