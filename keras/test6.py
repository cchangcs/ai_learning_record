import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# 使多次生成的随机数相同
np.random.seed(1337)

# 超参数
BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


# 生成数据
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

# 查看数据
# get_batch()
# exit()
# 搭建网络
model = Sequential()

# 添加LSTM层
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,
    return_sequences=True,  # 对于每一个时间点需不需要输出对应的output, True每个时刻都输出， False最后的输出output
    stateful=True,  # batch与batch之间是否有联系，需不需要将状态进行传递
))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))  # TimeDistributed：对每一个output进行全连接的计算

# 优化器
adam = Adam()
model.compile(
    optimizer=adam,
    loss='mse',
)

# 训练
print('Training ------------')
for step in range(501):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost: ', cost)