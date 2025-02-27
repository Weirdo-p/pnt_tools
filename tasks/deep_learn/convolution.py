import hashlib
import os
import tarfile
import zipfile
import requests

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys, os
from d2l import torch as d2l
sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger

def corr2d(X, K):
    h, w = K.shape

    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
logger.info(corr2d(X, K))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

X = torch.ones((6, 8))
X[:, 2:6] = 0
logger.info(X)

K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
logger.info(Y)
Y1 = corr2d(X.t(), K)
logger.info(Y1)

conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        logger.info(f'batch {i + 1}, loss {l.sum():.3f}')
logger.info(conv2d.weight.data.reshape((1, 2)))

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)

X = torch.rand(size=(8, 8))
logger.info(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
logger.info(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
logger.info(comp_conv2d(conv2d, X).shape)

def corr2d_multi_in(X, K):
    logger.info(X)
    logger.info(K)
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
corr2d_multi_in(X, K)

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

