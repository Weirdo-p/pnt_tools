import random, torch
import sys, os
from torch.distributions import multinomial
from d2l import torch as d2l

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger

#%% manual implementation
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w, true_b = torch.tensor([2, -3.4]), 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# logger.warning(features)
# d2l.set_figsize()
# d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1)
# d2l.plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) 
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    return X @ w + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr, num_epochs, net, loss = 0.03, 3, linreg, squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        logger.info(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

logger.info("===== summary of manually implementation ====")
logger.info(f"w的估计误差：{true_w - w.reshape(true_w.shape)}")
logger.info(f"b的误差：{true_b - b}")


features, labels = synthetic_data(true_w, true_b, 1000)

from torch.utils import data
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
logger.info(next(iter(data_iter)))

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.HuberLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
for epoch in range(5):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    logger.info(f'epoch {epoch + 1}, loss {l:f}')

w, b = net[0].weight.data, net[0].bias.data
logger.info(f"w的估计误差：{true_w - w.reshape(true_w.shape)}")
logger.info(f"b的误差：{true_b - b}")
