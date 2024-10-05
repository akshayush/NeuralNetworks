
from tokenize import Double
from numpy import float64
import torch.nn as nn
import torch
import math

class FirstNetwork_Parameters(nn.Module):
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.weights1 = nn.Parameter(torch.randn(2, 2) / math.sqrt(2))
    #self.weights1=self.weights1
    self.bias1 = nn.Parameter(torch.zeros(2))#.float
    self.weights2 = nn.Parameter(torch.randn(2, 4) / math.sqrt(2))#.float
    self.bias2 = nn.Parameter(torch.zeros(4))#.float
    
  def forward(self, X):
    a1 = torch.matmul(X.type(torch.float), self.weights1) + self.bias1
    h1 = a1.sigmoid()
    a2 = torch.matmul(h1, self.weights2) + self.bias2
    h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)
    return h2


class FirstNetwork_Linear_Optim(nn.Module):
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.lin1 = nn.Linear(2,2) ## Weights and biases of 2 of 2
    self.lin2 = nn.Linear(2,4)
    
    
  def forward(self, X):
    a1 = self.lin1(X)
    h1 = a1.sigmoid()
    a2 = self.lin2(h1)
    h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)
    return h2  


class FirstNetwork_Sequential(nn.Module):
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.net=nn.Sequential(
      nn.Linear(2,128),
      nn.Sigmoid(),
      nn.Linear(128,64),
      nn.Sigmoid(),
      nn.Linear(64,4),
      nn.Softmax()
      )
    
    
  def forward(self, X):
    return self.net(X)