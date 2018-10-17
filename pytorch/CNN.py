import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt


# 超参数
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False   # 已下载，设置为False,未下载，则设置为True

# 下载MNIST数据
# 训练数据
train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 数据保存地址
    train=True,  # 训练数据，False即为测试数据
    transform=torchvision.transforms.ToTensor(),  # 将下载的源数据变成Tensor数据，（0,1）
    download=DOWNLOAD_MNIST,
)

# 显示一张样本图片
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 测试数据
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000]/255.  # /255压缩数据区间为[0-1]
test_y = test_data.test_labels[:2000]


# 建立神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # shape(1, 28, 28)
                in_channels=1, # 高度
                out_channels=16,  # filter的数目
                kernel_size=5,  # filter 的宽度和高度
                stride=1,  # 步长
                padding=2,  # 如果stride=1,要使得经过conv之后与原来宽度一样，则padding=(kernel_size-1)/2=(5-1)/2=2
                # shape (16, 28, 28)
            ),  # 卷积层 filter
            nn.ReLU(),  # 激活函数 # shape (16, 28, 28)
            nn.MaxPool2d(  # shape (16, 14, 14)
                kernel_size=2
            ),  # 池化层
        )
        self.conv2 = nn.Sequential(  # shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # shape (32, 14, 14)
            nn.ReLU(),  # shape (32, 14, 14)
            nn.MaxPool2d(2)  # shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   # (batch , 32 , 7, 7)
        x = x.view(x.size(0), -1)  # （ batch, 32 * 7 * 7)
        output = self.out(x)
        return output


if __name__ == '__main__':
    cnn = CNN()

    # 打印网络结构
    # print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 优化CNN参数
    loss_func = nn.CrossEntropyLoss()  # 标签是one-hot形式的

    # 训练数据
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = np.squeeze(torch.max(test_output, 1)[1].data)
                accuracy = sum(pred_y == test_y) / test_y.size(0)
                print('Epoch: ', epoch, '   train loss: %.4f' % loss.data[0], '  test accuracy: %.2f' % accuracy)
                # 输出前10个测试数据的预测值
                # print 10 predictions from test data
    test_output = cnn(test_x[:10])
    pred_y = np.squeeze(torch.max(test_output, 1)[1].data.numpy())
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')
