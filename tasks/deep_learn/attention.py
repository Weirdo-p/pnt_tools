import torch
from torch import nn
from d2l import torch as d2l
import collections, re, os, sys
sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from tasks.deep_learn.RNN import load_data_time_machine, train
from utils.logger.logger import logger
from tasks.deep_learn.try_gpu import *


def f(x):
    return 2 * torch.sin(x) + x**0.8

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
    
if __name__ == "__main__":

    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)

    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
    x_test = torch.arange(0, 5, 0.1)
    y_truth = f(x_test)
    n_test = len(x_test)

    # X_repeat的形状:(n_test,n_train),
    # 每一行都包含着相同的测试输入（例如：同样的查询）
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # x_train包含着键。attention_weights的形状：(n_test,n_train),
    # 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
    attention_weights = nn.functional.softmax(-(X_repeat- x_train)**2 / 2, dim=1)
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
    y_hat = torch.matmul(attention_weights, y_train)
    # plot_kernel_reg(y_hat)
    logger.info(f"y_hat: {y_hat}")

    X_tile = x_train.repeat((n_test, 1))
    Y_tile = y_train.repeat((n_test, 1))
    # 使用注意力机制来计算新的预测值

    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) 
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        logger.info(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
