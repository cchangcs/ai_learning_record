import numpy as np
from keras import Sequential   # 按顺序建立的神经网络
from keras.layers import Dense  # Dense全连接层
import matplotlib.pyplot as plt

np.random.seed(1337)   # 使多次生成的随机数相同

# 生成数据
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  # 打乱生成的数据
Y = 0.5 * X + 2 + np.random.normal(0, 0.005, (200,))

# 显示生成的数据
# plt.scatter(X, Y)
# plt.show()

# 把数据分成训练数据和测试数据
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# 利用keras建造神经网络
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
# model.add(Dense(output_dim=1, ))   # 输入默认为上一层运行的输出

# 选择误差函数和优化方法
model.compile(loss='mse', optimizer='sgd')

# 训练
print('Training....')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost', cost)

# 测试
print('\nTesting....\n')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# 显示预测值
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()