import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

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
X_train = X_train.reshape(-1, 1, 28, 28)   # -1:sample个数， 1：channel, 28x28：长宽
X_test = X_test.reshape(-1, 1, 28, 28)
# 将标签变为one-hot形式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 搭建网络
model = Sequential()
# conv1 layer
model.add(Convolution2D(
    nb_filter=32,  # 滤波器
    nb_row=5,  # filter宽度
    nb_col=5,  # filter高度
    border_mode='same',  # padding的方法
    input_shape=(1,  # channel的个数
                 28, 28),  # width和height
))
model.add(Activation('relu'))
# 池化层 pooling
model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='same',  # padding method
))

# Conv2 layer
model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))

# pooling2 layer
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

# 展开
model.add(Flatten())
# 全连接层1
model.add(Dense(1024))
model.add(Activation('relu'))

# fc2
model.add(Dense(10))
model.add(Activation('softmax'))  # 用于分类的激活函数
# 优化器
adam = Adam(lr=1e-4)

# 激活模型
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练
print('Training...')
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

# 测试
print('\nTesing....\n')
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss', loss)
print('\ntest accuracy', accuracy)