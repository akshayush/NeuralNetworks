import torch.nn as nn
from torch.nn.modules import AvgPool1d

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.cnn_model=nn.Sequential(
            nn.Conv2d(3,6,5), # (N,3,32,32)--> (N,6,28,28)
            nn.Tanh(),
            nn.AvgPool2d(2,stride=2), #(N,6,28,28) --> (N,6,14,14)
            nn.Conv2d(6,16,5), #(N,6,14,14)--> (N,16,10,10)
            nn.Tanh(),
            nn.AvgPool2d(2,stride=2), #(N,16,10,10)--> (N,16,5,5)
            )
        self.fc_model=nn.Sequential(
            nn.Linear(400,120), #16*5*5 (400) --> 120
            nn.Tanh(),
            nn.Linear(120,84), # 120 --> 84
            nn.Tanh(),
            nn.Linear(84,10)
        )
    def forward(self,x):
        #print(x.shape)
        x=self.cnn_model(x)
        #print(x.shape)
        x=x.view(x.size(0),-1)
        #print(x.shape)
        x=self.fc_model(x)
        #print(x.shape)
        return x
    