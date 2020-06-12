#_*_coding:utf-8_*_
'''
@project: work
@author: exudingtao
@time: 2020/6/5 2:43 下午
'''

import torch
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
#定义训练集和labels
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0]*features[:, 0] + true_w[1]*features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
dataset = Data.TensorDataset(features, labels)
dataloader = Data.DataLoader(dataset, batch_size, shuffle=True)


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    #forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
##写法2
net_1 = nn.Sequential(
    nn.Linear(num_inputs, 1)
    #还可以定义其他层
)
##写法3
net_2 = nn.Sequential()
net_2.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......
# 写法4
from collections import OrderedDict
net_3 = nn.Sequential(
    OrderedDict([
        ('linear', nn.Linear(num_inputs, 1))
        # .....
    ])
)
#通过输出net.parameters()来查看模型所有的可学习参数
for param in net.parameters():
    print(param)


#使用net前，需要初始化模型参数，通过init
# from torch.nn import init
# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)
#损失函数
loss = nn.MSELoss()
#优化算法
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
#还可以为不不同⼦子⽹网络设置不不同的学习率，这在finetune时经常⽤用到。例例:
# optimizer =optim.SGD([# 如果对某个参数不不指定学习率，就使⽤用最外层的默认学习率
#     {'params': net.subnet1.parameters()},  # lr=0.03
#     {'params': net.subnet2.parameters(), 'lr': 0.01}
#     ], lr=0.03)

#训练
num_epochs = 3
for epoch in range(1, num_epochs+1):
    for x,y in dataloader:
        output = net(x)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

