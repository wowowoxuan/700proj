import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import h5py
import pdb
from tqdm import tqdm, trange


class PermEqui_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x

class PermEqui_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)

    xm = self.Gamma(xm)
    x = self.Lambda(x)
    x = x - xm
    return x

class Deepset(nn.Module):
  def __init__(self, perm_layer_type = 'max', fg = False, d_dim, x_dim=26):
    super(Deepset, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim
    self.perm_layer_type = perm_layer_type
    self.fg = fg
    if self.perm_layer_type == 'max':
      self.perm = nn.Sequential(
        PermEqui_max(self.x_dim, self.d_dim),
        nn.Tanh(),
        PermEqui_max(self.d_dim, self.d_dim), 
        nn.Tanh(),    
      )
      self.lastlayer = PermEqui_max(self.d_dim, self.x_dim)
    else:
      self.perm = nn.Sequential(
        PermEqui_mean(self.x_dim, self.d_dim),
        nn.Tanh(),
        PermEqui_mean(self.d_dim, self.d_dim),  
        nn.Tanh(),  
      )
      self.lastlayer = PermEqui_mean(self.d_dim, self.x_dim) 

    self.down_samplingpool = nn.MaxPool1d(2,stride = 2)
    

  def forward(self, x):
    x = self.perm_layer_type(x)
    if fg:
      x = self.transpose(1,2)
      x = self.down_samplingpool(x)
      x = self.transpose(1,2)
    x = self.lastlayer(x)
    


    return x


