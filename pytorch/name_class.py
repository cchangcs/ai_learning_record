# ===========================准备数据================================
from IPython.core.interactiveshell import InteractiveShell
import glob

InteractiveShell.ast_node_interactivity = 'all'
# *是通配符，匹配出data文件夹下的所有txt文件
all_filenames = glob.glob('data/*.txt')
# print(all_filenames)

# ===========================转换编码================================
import unicodedata
import string
# 姓氏中所有的字符
# string.ascii_letters是大小写各26字符
all_letters = string.ascii_letters + '.,;'
# 字符的种类数
n_letters = len(all_letters)


# 将Unicode编码转换成标准的ASCII码
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# print(n_letters)  # 字符数为57个
# print(unicode_to_ascii('Ślusàrski'))

# ===========================读取数据================================
category_names = {}
all_categories = []


# 读取txt文件，返回ascii码的姓名、列表
def readNames(filename):
    names = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


for filename in all_filenames:
    category = filename.split('\\')[-1].split('.')[0]
    all_categories.append(category)
    names = readNames(filename)
    category_names[category] = names

# 语言种类数
n_categories = len(all_categories)
# print('n_categories = ', n_categories)
# print(category_names['Italian'][:5])

# ===========================将数据转换为Tensor================================
import torch


# 将字符转化为 < 1 x n_letters> 的Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor


# 将姓名转化为尺寸为<name_length x 1 x n_letters>的数据
# 使用的是one-hot编码方式转化
def name_to_tensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for ni, letter in enumerate(name):
        letter_index = all_letters.find(letter)
        tensor[ni][0][letter_index] = 1
    return tensor

# print(letter_to_tensor('J'))
# ===========================构建神经网络================================
'''
input_size: 表征字母的向量的特征数量（向量长度）
hidden_size: 隐藏层特征数量（列数）
output_size: 语言数目，18
i2h: 隐藏网络参数的计算过程。输入的数据尺寸为input_size + hidden_size, 输出的尺寸为 hidden_size
i2o: 输出网络参数的计算过程。输入的数据尺寸为input_size + hidden_size, 输出的尺寸为 output_size
'''
import torch.nn as nn
from torch.autograd import Variable
# a = torch.zeros(3, 1)
# b = torch.zeros(3, 2)
# print(a)
# print(b)
# print(torch.cat((a, b), 1))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        # 将input和之前的网络中的隐藏层参数合并。
        combined = torch.cat((input, hidden), 1)

        hidden = self.i2h(combined)  # 计算隐藏层参数
        output = self.i2o(combined)  # 计算网络输出的结果
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        # 初始化隐藏层参数hidden
        return torch.zeros(1, self.hidden_size)

# 创建RNN的实例
rnn = RNN(
    input_size=n_letters,    # 输入每个字母向量的长度（57个字符）
    hidden_size=128,  # 隐藏层向量的长度，神经元个数。这里可自行调整参数大小
    output_size=n_categories    # 语言的种类数目
)


# input = letter_to_tensor('A')
# hidden = rnn.init_hidden()
# output, next_hidden = rnn(Variable(input), Variable(hidden))
# print('output = ', output.size())

# input = name_to_tensor('Albert')
# hidden = rnn.init_hidden()  # #这里的128是hidden_size
#
# # 给rnn传入的初始化hidden参数是尺寸为（1， 128）的zeros矩阵
# # input[0]是传入姓名的第一个字符数组，注意这个数组是batch_size=1的矩阵。因为在pytorch中所有输入的数据都是batc
# output, next_hidden = rnn(Variable(input[0]), Variable(hidden))
# print(output.size())
# print(output)
#
# print(output.data)
# print(output.data.topk(1))


# ===========================准备训练RNN================================
# 求所属语言类别的索引值
def category_from_output(output):
    _, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i],category_i

# print(category_from_output(output))
# 增入随机性
import random


def random_training_pair():
    category = random.choice(all_categories)
    name = random.choice(category_names[category])
    category_tensor = torch.LongTensor([all_categories.index(category)])
    name_tensor = name_to_tensor(name)
    return category, name, category_tensor, name_tensor

# 从数据集中抽取十次
# for i in range(10):
#     category, name, category_tensor, name_tensor = random_training_pair()
#     print('category = ', category, '  name = ', name)
# ===========================训练RNN网络================================
# loss function
loss_func = nn.CrossEntropyLoss()
# optimizer
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(),
                            lr=learning_rate)


# 训练
def train(category_tensor, name_tensor):
    rnn.zero_grad()  # 将rnn网络梯度清零
    hidden = rnn.init_hidden()  # 只对姓名的第一字母构建起hidden参数

    # 对姓名的每一个字母逐次学习规律。每次循环的得到的hidden参数传入下次rnn网络中
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)

    # 比较最终输出结果与 该姓名真实所属语言，计算训练误差
    loss = loss_func(output, category_tensor)
    # 将比较后的结果反向传播给整个网络
    loss.backward()

    # 调整网络参数。有则改之无则加勉
    optimizer.step()

     # 返回预测结果  和 训练误差
    return output, loss.data[0]

# 练100000次
import time
import math

n_epoches = 100000  # 训练100000次（可重复的从数据集中抽取100000姓名）
print_every = 5000  # 每训练5000次，打印一次
plot_every = 1000   # 每训练1000次，计算一次训练平均误差

current_loss = 0  # 初始误差为0
all_losses = []  # 记录平均误差


def time_since(since):
    # 计算训练使用的时间
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 训练开始时间点
start = time.time()

for epoch in range(1, n_epoches + 1):
    # 随机的获取训练数据name和对应的language
    category, name, category_tensor, name_tensor = random_training_pair()
    output, loss = train(Variable(category_tensor), Variable(name_tensor))
    current_loss += loss

    # 每训练5000次，预测一个姓名，并打印预测情况
    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (
        epoch, epoch / n_epoches * 100, time_since(start), loss, name, guess, correct))

    # 每训练5000次，计算一个训练平均误差，方便后面可视化误差曲线图
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# ===========================测试================================

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)
plt.show()

def predict(rnn, input_name, n_predictions=3):
    hidden = rnn.init_hidden()

    # name_tensor.size()[0] 名字的长度(字母的数目)
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], Variable(hidden))
    print('\n> %s' % input_name)

    # 得到该姓名预测结果中似然值中前n_predictions大的 似然值和所属语言
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []
    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])


predict(rnn, 'Dovesky')
predict(rnn, 'Jackson')
predict(rnn, 'Satoshi')