import numpy as np
import torch
import math
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy.random as nr
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F

## NN parameters Initialization 
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

## Define NNs here
class Net(nn.Module):
    def __init__(self, X_dim, Y_dim, out_dim, hidden1=400, hidden2=300, init_w=3e-3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_dim, hidden1)
        self.fc2 = nn.Linear(hidden1+Y_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Find k_neighbors-nearest neighbor distances from y for each example in a minibatch x.
def knn(x, y, k=1, last_only=False, discard_nearest=True):
    
    
    dist_x = (x ** 2).sum(-1).unsqueeze(1)  # [T, 1]
    dist_y = (y ** 2).sum(-1).unsqueeze(0)  # [1, T']
    cross = - 2 * torch.mm(x, y.transpose(0, 1))  # [T, T']
    distmat = dist_x + cross + dist_y  # distance matrix between all points x, y
    distmat = torch.clamp(distmat, 1e-8, 1e+8)  # can have negatives otherwise!

    if discard_nearest:  # never use the shortest, since it can be the same point
        knn, _ = torch.topk(distmat, k + 1, largest=False)
        knn = knn[:, 1:]
    else:
        knn, _ = torch.topk(distmat, k, largest=False)

    if last_only:
        knn = knn[:, -1:]  # k_neighbors:th distance only

    return torch.sqrt(knn)


def kl_div(x, y, k=1, eps=1e-8, last_only=False):
    """KL divergence estimator for batches x~p(x), y~p(y).
    :param x: prediction; shape [T, N]
    :param y: target; shape [T', N]
    :param k:
    :return: scalar
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x.astype(np.float32))
        y = torch.tensor(y.astype(np.float32))

    nns_xx = knn(x, x, k=k, last_only=last_only, discard_nearest=True)
    nns_xy = knn(x, y, k=k, last_only=last_only, discard_nearest=False)

    divergence = (torch.log(nns_xy + eps) - torch.log(nns_xx + eps)).mean()

    return divergence


def entropy(x, k=1, eps=1e-8, last_only=False):
    """Entropy estimator for batch x~p(x).
        :param x: prediction; shape [T, N]
        :param k:
        :return: scalar
        """
    if type(x) is np.ndarray:
        x = torch.tensor(x.astype(np.float32))

    nns_xx = knn(x, x, k=k, last_only=last_only, discard_nearest=True)

    ent = torch.log(nns_xx + eps).mean() - torch.log(torch.tensor(eps))

    return ent

#Uniform Slices
def rand_slices(dim, num_slices,high=10,low=1):
    
    slices = (high - low) * torch.rand((num_slices, dim))+ low
    slices = slices / torch.sqrt(torch.sum(slices ** 2, dim=1, keepdim=True))
    return slices


# arccos must be \geq \omega_x (resp. \omega_y)
def arccos_distance_torch(x1, x2, eps=1e-7):
    # x2 = x1 if x2 is None else x2
    
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    
    return torch.mean(torch.acos(torch.clamp(torch.abs(torch.matmul(x1, x2.t())/ (w1 * w2.t()).clamp(min=eps)),min=-1+eps,max=1-eps)))

# KSG Estimator   
def I_est(X,Y):
    # print('g')
    # return continuous.get_mi(x,y)
    
    return entropy(Y)+entropy(X)-entropy(torch.cat((X,Y),dim=1))