#_*_coding:utf-8_*_
'''
@project: work
@author: exudingtao
@time: 2020/6/11 4:17 下午
'''
import torch
import numpy as np
import sys
sys.path.append("..")
import matplotlib.pyplot as plt

#⽣生成数据集
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)#按列拼接
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)#噪声项 服从均值为0、标准差为0.01的正态分布
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
#多项式函数拟合也使⽤用平⽅方损失函数
num_epochs, loss = 100, torch.nn.MSELoss()
def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1) #pytorch已经将参数初始化了了，所以我们这⾥里里就不不⼿手动初始 化了了
    batch_size = min(10, train_labels.shape[0])
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_dataloader:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch:train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('wight:', net.weight.data, '\nbias:', net.bias.data)
#函数拟合
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
#欠拟合
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])
#过拟合 （训练样本不足）
fit_and_plot(poly_features[0:5, :], poly_features[n_train:, :], labels[0:5], labels[n_train:])
