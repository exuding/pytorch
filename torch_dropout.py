#_*_coding:utf-8_*_
'''
@project: work
@author: exudingtao
@time: 2020/6/12 11:54 上午
'''

import torch
import numpy as np
import torch.nn as nn


def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()
    return mask * X / keep_prob


X = torch.arange(16).view(2, 8)
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1.0))

'''
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
tensor([[ 0.,  0.,  4.,  6.,  8., 10.,  0.,  0.],
        [ 0.,  0., 20., 22.,  0.,  0., 28., 30.]])
tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
'''

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]
#通常的建议是把靠近输⼊入层的丢弃概率设得⼩小⼀一点
#第⼀一个隐藏层的丢弃概率设为0.2，把第⼆二个隐藏层的丢弃概率设为0.5
drop_prob1, drop_prob2 = 0.2, 0.5
#模型将全连接层和激活函数ReLU串起来，并对每个激活函数的输出使⽤丢弃法
def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training: # 只在训练模型时使⽤用丢弃法
        H1 = dropout(H1, drop_prob1) # 在第⼀一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2) # 在第⼆二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3

#模型评估
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()  # 改回训练模式
        else:  # ⾃自定义的模型
            if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

#训练和测试模型
num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = load_data_fashion_mnist(batch_size)#之前定义的
train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size, params, lr)#之前定义的


#---------------pytorch实现-----------------#
'''
在PyTorch中，我们只需要在全连接层后添加Dropout层并指定丢弃概率
在训练模型时， Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素;
在测试模型时(即 model.eval()后，Dropout层并不发挥作用。
'''
net = nn.Sequential(
    FlattenLayer(),#前面定义的
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, 10)
    )
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)





