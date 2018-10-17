import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 生成假数据
# torch.unsqueeze() 的作用是将一维变二维，torch只能处理二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape(100, 1)
# 0.2 * torch.rand(x.size())增加噪点
y = x.pow(2) + 0.2 + 0.2 * torch.rand(x.size())

# 将Tensor转换为torch
x, y = Variable(x), Variable(y)

# 打印数据散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    # 初始化
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 前向传递
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
# 输出定义的网络的结构
print(net)
plt.ion()
plt.show()

# 优化(给出神经网络的参数和学习速率)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# loss function，回归问题：均反差（MSE）
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)
    # 求误差
    loss = loss_func(prediction, y)

    # 优化
    # 每一步首先将梯度降为0
    optimizer.zero_grad()
    # 进行反向传递更新参数
    loss.backward()
    # 优化梯度
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
