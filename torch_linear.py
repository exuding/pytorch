#_*_coding:utf-8_*_
'''
@project: work
@author: exudingtao
@time: 2020/6/5 12:18 下午
'''

import torch
import numpy as np
import random

#定义训练集和labels
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
labels = true_w[0]*features[:, 0] + true_w[1]*features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))


# 本函数已保存在d2lzh包中⽅方便便以后使⽤用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size, num_examples)]) # 最后⼀一次可能不不⾜足⼀一个batch
        yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
#初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
#模型训练中，需要对这些参数求梯度来迭代参数的值
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


#定义模型
def linreg(x, w, b):
    x = torch.tensor(x, dtype=torch.float32)
    return torch.mm(x, w) + b


#loss
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

#优化更新参数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这⾥里里更更改param时⽤用的param.data
        #除以批量量⼤大⼩小来得到平 均值


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        out = net(x, w, b)
        l = loss(out, y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        #梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_loss = loss(net(features, w, b,), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_loss.mean().item()))

