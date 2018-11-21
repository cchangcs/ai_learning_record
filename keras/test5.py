import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.optimizers import Adam

# 使多次生成的随机数相同
np.random.seed(1337)

# 超参数
TIME_STEP = 28  # 和图片的高度相同
INPUT_SIZE = 28  # 和图片的宽度相同
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10  # [0 0 0 0 1 0 0 0 0 0]->4
CELL_SIZE = 50  # RNN里面hidden unit的数量
LR = 0.001

# 下载数据集
# X_shape(60000 28x28),y shape(10000)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 预处理数据
'''
X_train.reshape(X_train.shape[0], -1) 将60000个28x28的数据变为60000x784
/255：把数据标准化到[0,1]
'''
# 除以255为进行标准化
X_train = X_train.reshape(-1, 28, 28) / 255   # -1:sample个数， 1：channel, 28x28：长宽
X_test = X_test.reshape(-1, 28, 28) / 255
# 将标签变为one-hot形式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    batch_input_shape=(None, TIME_STEP, INPUT_SIZE),
    output_dim=CELL_SIZE,
))
# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))  # softmax进行分类

# 优化器
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练
for step in range(4001):
    # data shape = (batch_num, steps, input/output)
    x_batch = X_train[BATCH_INDEX: BATCH_SIZE + BATCH_INDEX, :, :]
    y_batch = y_train[BATCH_INDEX: BATCH_SIZE + BATCH_INDEX, :]
    cost = model.train_on_batch(x_batch, y_batch)

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost:', cost, 'test accuracy:', accuracy)