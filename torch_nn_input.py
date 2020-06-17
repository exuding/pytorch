#_*_coding:utf-8_*_
'''
@project: work
@author: exudingtao
@time: 2020/6/17 4:46 下午
'''
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import random
import zipfile
sys.path.append("..")


#-----------------------------输入----------------------------#
def get_data(file_path):
    #读取数据集
    with zipfile.ZipFile(file_path) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars[:40]
    #个数据集有6万多个字符。为了打印⽅便，我们把换⾏符替换成空格，然后仅使⽤前1万个字符来训练模型
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    #建立字符索引
    '''
    将每个字符映射成⼀个从0开始的连续整数，⼜称索引，来⽅便之后的数据处理
    为了得到索引，我们将数据集⾥所有不同字符取出来，然后将其逐⼀映射到索引来构造词典
    '''
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    #将训练数据集中每个字符转化为索引
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


#随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输⼊的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的⻓为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


# my_seq = list(range(30))
# for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')


#相邻采样
#除对原始序列做随机采样之外，我们还可以令相邻的两个随机⼩批量在原始序列上的位置相毗邻
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0:batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

#
# for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')

