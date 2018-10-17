import torch
import torch.utils.data as Data

BATCH_SIZE = 8
x = torch.linspace(1, 10, 10)  # x (torch tensor)
y = torch.linspace(10, 1, 10)  # y (torch tensor)

torch_dataset = Data.TensorDataset(x, y)

# Loader让数据变成多个小批次
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 在训练时需不需要打乱数据后再进行抽样
    # num_workers=2,  # 提取的时候使用两个线程来进行提取
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        # training
        print('Epoch: ', epoch, '| step: ', step, '| bacth_x: ',
              batch_x.numpy(), '| batch_y: ', batch_y.numpy())
