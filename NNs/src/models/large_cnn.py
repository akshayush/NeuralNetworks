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
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import make_blobs
from torch import optim
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import copy

def inceptionv3_model():
  net=torchvision.models.inception_v3(pretrained=True)
  for param in net.parameters():
    param.requires_grad=False
  aux_in_features = net.AuxLogits.fc.in_features
  net.AuxLogits.fc=nn.Linear(aux_in_features,10)
  in_features = net.fc.in_features
  net.fc=nn.Linear(in_features,10)
  print(net)
  for param in net.parameters():
    if param.requires_grad:
      print(param.shape)
  return net

def resnet18_model():
  net=torchvision.models.resnet18(pretrained=True)
  for param in net.parameters():
    param.requires_grad=False
  final_in_features = net.fc.in_features
  net.fc=nn.Linear(final_in_features,10)
  print(net)
  for param in net.parameters():
    if param.requires_grad:
      print(param.shape)
  return net


def vgg_model():
  net=torchvision.models.vgg16_bn(pretrained=True)
  for param in net.parameters():
    param.requires_grad=False
  final_in_features = net.classifier[6].in_features
  net.classifier[6]=nn.Linear(final_in_features,10)
  print(net)
  for param in net.parameters():
    if param.requires_grad:
      print(param.shape)
  return net

def large_cnn_fit(trainloader,testloader,network,device,model="other",lr=0.05,batch_size=16, epochs=1):
  loss_fn=nn.CrossEntropyLoss()
  opt=torch.optim.Adam(network.parameters(),lr)
  loss_arr=[]
  min_loss=1000
  n_iters=np.ceil(50000/batch_size)
  loss_epoch_arr=[]
  for epoch in range(epochs):
    for i, data in enumerate(trainloader,0):
      inputs,labels=data
      if device:
        inputs,labels=inputs.to(device),labels.to(device)        
      opt.zero_grad()
      
      if model=="Inception":
        outputs,aux_outputs=network(inputs)
      else:
        outputs=network(inputs)
      if model=="Inception":
        loss=loss_fn(outputs,labels) + 0.3 * loss_fn(aux_outputs,labels)
      else:
        loss=loss_fn(outputs,labels)
      loss.backward()
      opt.step()
      if min_loss>loss.item():
        min_loss=loss.item()
        best_model=copy.deepcopy(network.state_dict())
        print("Min Loss %0.2f" %min_loss)
      del inputs,labels,outputs
      torch.cuda.empty_cache()
      if i % 100==0:
        print('Iteration: %d/%d , Loss : %0.2f' % (i,n_iters,loss.item()))
      loss_arr.append(loss.item())
    loss_epoch_arr.append(loss.item())
    print('Epoch: %d/%d, test Acc: %0.2f, Train acc: %0.2f'%(epoch,epochs,gn_evaluation(testloader,network,device,model),gn_evaluation(trainloader,network,device,model)))
  plt.plot(loss_epoch_arr)
  plt.show()


def CIFAR10_data_transformed(batchsize=128,size=224):
  transform_train=transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
  transform_test=transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
  trainset=torchvision.datasets.CIFAR10(
    root="C:\\Users\\aksha\\source\\repos\\NNs\\NNs\\data\\",
    train=True,download=True,transform=transform_train)
  trainloader=torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True)
  testset=torchvision.datasets.CIFAR10(root="C:\\Users\\aksha\\source\\repos\\NNs\\NNs\\data\\",
                                       train=False,download=True,transform=transform_test)
  testloader=torch.utils.data.DataLoader(testset,batch_size=batchsize,shuffle=False)
  return trainloader,testloader
  

def gn_evaluation(dataloader,net,device,model="other"):
  total,correct=0,0
  for data in dataloader:
    inputs,labels=data
    if device:
      inputs,labels=inputs.to(device),labels.to(device)
    if model=="Inception":
      outputs,aux_outputs=net(inputs)
    else:
      outputs=net(inputs)
    _,pred=torch.max(outputs.data,1)
    total+=labels.size(0)
    correct+=(pred==labels).sum().item()
  return 100*correct/total