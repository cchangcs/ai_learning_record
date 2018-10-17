import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 生成假数据
# torch.unsqueeze() 的作用是将一维变二维，torch只能处理二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape(100, 1)
# 0.2 * torch.rand(x.size())增加噪点
y = x.pow(2) + 0.2 + 0.2 * torch.rand(x.size())

# 将Tensor转换为torch
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

# 保存神经网络


def save():
    # 搭建神经网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # 优化器:随机梯度下降
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)

    # 损失函数：均方差
    loss_func = torch.nn.MSELoss()

    # 训练100步
    for i in range(100):
        prediction = net1(x)
        # 求误差
        loss = loss_func(prediction, y)
        # 优化
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
    # 绘图
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    # 保存
    torch.save(net1, 'net.pkl')  # save entire net
    torch.save(net1.state_dict(), 'net_params.pkl')   # save parameters


# 提取神经网络：保存整个网络的形式
def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    # 绘图
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


# 提取神经网络：保存网络参数的方式
# 这种方式速度更快
def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    # 绘图
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

save()

restore_net()

restore_params()