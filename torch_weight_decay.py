#_*_coding:utf-8_*_
'''
@project: work
@author: exudingtao
@time: 2020/6/11 5:08 下午
'''

#应对过拟合问题的常⽤用⽅方法:权重衰减(weight decay)
#权􏰁重衰减等价于L2范数正则化(regularization)
#高维线性回归为例例来引⼊入⼀一个过拟合问题，并使⽤用权􏰁衰减来应对过拟合


import torch
import numpy as np
import torch.nn as nn
import sys
sys.path.append("..")
import matplotlib.pyplot as plt

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)#噪音项
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


#随机初始化模型参数，给梯度
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


#l2范数惩罚项
def l2_penalty(w):
    return (w**2).sum()/2


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


#定义作图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None):
    plt.figure(num=1, figsize=(8, 4))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


#训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = linreg, squared_loss


dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot2(lambd):
    #初始化参数
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()
            if w.grad is not None:
                w.grad.data.zero_()#参数梯度置零
                b.grad.data.zero_()
            l.backward()
            sgd([w, b], lr, batch_size)#优化更新参数
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    #semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())


#设置lambd=0时，没有使用权重衰减，结果训练误差远小于测试集上的误差，这是典型的过拟合现象
# fit_and_plot2(lambd=0)
# fit_and_plot2(lambd=3)#此时权重参数更接近于0


'''
正则化通过为模型损失函数添加惩罚项使学出的模型参数值较⼩小，是应对过拟合的常⽤用⼿手段。 
权􏰁衰减等价于l2范数正则化，通常会使学到的权􏰁参数的元素较接近0。 
权􏰁重衰减可以通过优化器器中的weight_decay超参数来指定。 
可以定义多个优化器器实例例对不不同的模型参数使⽤用不不同的迭代⽅方法
'''
def fit_and_plot_pytorch(wd):
    # 对权􏰁参数衰减。权􏰁名称⼀一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd) # 对权􏰁参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            # 对两个optimizer实例例分别调⽤用step函数，从⽽而分别更更新权􏰁和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    #semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())

fit_and_plot_pytorch(0)
fit_and_plot_pytorch(3)
