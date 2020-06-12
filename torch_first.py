#_*_coding:utf-8_*_
'''
@project: work
@author: exudingtao
@time: 2020/5/29 10:49 上午
'''

import torch
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import torch.utils.data as Data
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from time import time


if __name__ == '__main__':
    pass
    ###-----------------------##------------------------###
    # target = torch.tensor([1, 0])
    # input = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float)
    # inputs = torch.randn(3, 5, requires_grad=True)
    # a = F.nll_loss(input, target, reduction='none')
    # b = F.log_softmax(inputs)
    # print(inputs)
    # print(b)
    ###-----------------------##------------------------###
    # a = torch.tensor([1., 2, 3, 4])
    # b = torch.tensor([4., 5, 6, 7])
    # loss_fn = nn.CrossEntropyLoss()
    # loss = loss_fn(a, b)
    ###-----------------------##------------------------###
    entroy = nn.CrossEntropyLoss()
    input = torch.Tensor([[0.5, 0.2, 0.3]])
    target = torch.tensor([1])
    output = entroy(input, target)
    print(output)
    #tensor(1.2398)

    ###-----------------------##------------------------###
    # a = torch.tensor([1., 2, 3, 4])
    # b = torch.tensor([1.1, 5, 6, 7])
    # loss_fn = nn.SmoothL1Loss(reduction='none')
    # loss = loss_fn(a, b)
    # print(loss)
    # out
    ###-----------------------##------------------------###
    # x = torch.tensor([1, 2])
    # y = torch.tensor([3, 4])
    # id_before = id(y)
    # #torch.add(x, y, out=y)  # y += x, y.add_(x)
    # y.add_(x)
    # print(id(y) == id_before)  # True
    ###-----------------------##------------------------###
    # x = torch.tensor(1.0, requires_grad=True)
    # y1 = x ** 2
    # with torch.no_grad():
    #     y2 = x ** 3
    # y3 = y1 + y2
    # print(x.requires_grad)
    # print(y1, y1.requires_grad)  # True 9 print(y2, y2.requires_grad) # False
    # print(y2, y2.requires_grad)  # False
    # print(y3, y3.requires_grad)  # True
    # y3.backward()#y3对x求梯度
    # print(x.grad)
    ###-----------------------##------------------------###
    # x = torch.tensor(1.0, requires_grad=True)
    # print(x.data)# 还是⼀一个tensor
    # print(x.data.requires_grad)# 但是已经是独⽴立于计算图之外,不会影响后面的依赖x的对x的梯度
    # y = 2* x
    # x.data *= 100
    # y.backward()#y对x求梯度
    # print(x)# 但是 更更改data的值也会影响tensor的值
    # print(x.grad)
    ###-----------------------##------------------------###
    # a = torch.ones(1000)
    # b = torch.ones(1000)
    # start = time()
    # c = torch.zeros(1000)
    # for i in range(1000):
    #     c[i] = a[i] + b[i]
    # print(time() - start)
    # start = time()
    # d = a + b #矢量计算更快
    # print(time()-start)
    ###-----------------------##------------------------###
    # a = torch.ones(3)
    # b = 10
    # print(a+b) #数值参与向量的矢量计算触发了广播机制
    ###-----------------------##------------------------###





