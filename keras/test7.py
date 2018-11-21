import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# 使多次生成的随机数相同
np.random.seed(1337)

# 获取数据
(x_train, _), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255. - 0.5       # 标准化
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

# 为了显示图片
encoding_dim = 2

# input placeholder
input_img = Input(shape=(784, ))

# encoder layer
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)

# decoder layer
decoded = Dense(10, activation='relu')(encoded_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded_output = Dense(784, activation='tanh')(decoded)

# AutoEncoder model
autoencoder = Model(inputs=input_img, outputs=decoded_output)

# Encoder Model
encoder = Model(inputs=input_img, outputs=encoded_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# 训练
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True)

# 显示
encoded_img = encoder.predict(x_test)
plt.scatter(encoded_img[:, 0], encoded_img[:, 1], c=y_test)
plt.colorbar()
plt.show()