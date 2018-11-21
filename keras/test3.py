import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# 使多次生成的随机数相同
np.random.seed(1337)

# 下载数据集
# X_shape(60000 28x28),y shape(10000)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 预处理数据
'''
X_train.reshape(X_train.shape[0], -1) 将60000个28x28的数据变为60000x784
/255：把数据标准化到[0,1]
'''
X_train = X_train.reshape(X_train.shape[0], -1) / 255  # 标准化
X_test = X_test.reshape(X_test.shape[0], -1) / 255  # 标准化
# 将标签变为one-hot形式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 搭建神经网络
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),   # 默认上一层的输出为本层输入
    Activation('softmax'),
])

# 定义Optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

# 激活模型
model.compile(
    optimizer=rmsprop,   # optimizer='rmsprop'使用默认的rmsprop优化函数
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# 训练
print('Training......')
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

# 测试
print('\nTesting......')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss', loss)
print('test accuracy', accuracy)