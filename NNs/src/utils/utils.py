
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




def CIFAR10_data(batchsize=128):
  trainset=torchvision.datasets.CIFAR10(root="C:\\Users\\aksha\\source\\repos\\NNs\\NNs\\data\\",train=True,download=True,transform=torchvision.transforms.ToTensor())
  trainloader=torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True)
  testset=torchvision.datasets.CIFAR10(root="C:\\Users\\aksha\\source\\repos\\NNs\\NNs\\data\\",train=False,download=True,transform=torchvision.transforms.ToTensor())
  testloader=torch.utils.data.DataLoader(testset,batch_size=batchsize,shuffle=False)
  return trainloader,testloader




def cnn_evaluation(dataloader,net,device):
  total,correct=0,0
  for data in dataloader:
    inputs,labels=data
    if device:
      inputs,labels=inputs.to(device),labels.to(device)
    outputs=net(inputs)
    _,pred=torch.max(outputs.data,1)
    total+=labels.size(0)
    correct+=(pred==labels).sum().item()
  return 100*correct/total


def cnn_fit(trainloader,testloader,network, epochs=16):
  loss_fn=nn.CrossEntropyLoss()
  opt=torch.optim.Adam(network.parameters())
  loss_arr=[]
  loss_epoch_arr=[]
  for epoch in range(epochs):
    for i, data in enumerate(trainloader,0):
      inputs,labels=data
      opt.zero_grad()
      outputs=network(inputs)
      loss=loss_fn(outputs,labels)
      loss.backward()
      opt.step()
      loss_arr.append(loss.item())
    loss_epoch_arr.append(loss.item())
    print('Epoch: %d/%d, test Acc: %0.2f, Train acc: %0.2f'%(epoch,epochs,cnn_evaluation(testloader,network),cnn_evaluation(trainloader,network)))
  plt.plot(loss_epoch_arr)
  plt.show()

def accuracy(y_hat, y):
  pred = torch.argmax(y_hat, dim=1)
  return (pred == y).float().mean()

def get_test_train_data(random_state :int=0,is_show=False):
  data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=random_state)
  print(data.shape, labels.shape)
  my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])
  plt.scatter(data[:,0], data[:,1], c=labels, cmap=my_cmap)
  if is_show:
    plt.show()
  X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)
  print(X_train.shape, X_val.shape, labels.shape)
  X_train, Y_train, X_val, Y_val = map(torch.tensor, (X_train, Y_train, X_val, Y_val))
  return X_train.type(torch.float), X_val, Y_train.type(torch.LongTensor), Y_val

def fit_optim(X_train,Y_train,fn,epochs = 1000, learning_rate = 1):
  loss_arr = []
  acc_arr = []
  opt=optim.SGD(fn.parameters(),lr=learning_rate)
  for epoch in range(epochs):
    y_hat = fn(X_train)
    loss = F.cross_entropy(y_hat, Y_train)
    loss_arr.append(loss.item())
    acc_arr.append(accuracy(y_hat, Y_train))

    loss.backward()
    opt.step()
    opt.zero_grad()
        
  plt.plot(loss_arr, 'r-')
  plt.plot(acc_arr, 'b-')
  plt.show()      
  print('Loss before training', loss_arr[0])
  print('Loss after training', loss_arr[-1])


def fit(X_train,Y_train,fn,epochs = 1000, learning_rate = 1):
  loss_arr = []
  acc_arr = []
  for epoch in range(epochs):
    y_hat = fn(X_train)
    loss = F.cross_entropy(y_hat, Y_train)
    loss_arr.append(loss.item())
    acc_arr.append(accuracy(y_hat, Y_train))

    loss.backward()
    with torch.no_grad():
      for param in fn.parameters():
        param -= learning_rate * param.grad
      fn.zero_grad()
        
  plt.plot(loss_arr, 'r-')
  plt.plot(acc_arr, 'b-')
  plt.show()      
  print('Loss before training', loss_arr[0])
  print('Loss after training', loss_arr[-1])


def fit_v3(x,y,model,opt,loss_fn,epochs=1000):
  for epoch in range(epochs):
    loss=loss_fn(model(x),y)
    loss.backward()
    opt.step()
    opt.zero_grad()
  return loss.item()