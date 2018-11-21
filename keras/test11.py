from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
'''
我们有404个训练样本和102个测试样本。 该数据包括13个功能。 输入数据中的13个功能如下
＃1。人均犯罪率。
＃2。占地面积超过25,000平方英尺的住宅用地比例。
＃3。每个城镇非零售业务占比的比例。
＃4。Charles River虚拟变量（如果管道限制河流则= 1;否则为0）。
＃5。一氧化氮浓度（每千万份）。
＃6。每个住宅的平均房间数。
＃7。1940年以前建造的自住单位比例。
＃8。到波士顿五个就业中心的加权距离。
＃9。径向高速公路的可达性指数。
＃10。每10,000美元的全额物业税率。
＃11。城镇的学生与教师比例。
＃12. 1000 *（Bk - 0.63）** 2其中Bk是城镇黑人的比例。
＃13.人口状况较低。
＃
＃目标是自住房屋的中位数，以千美元计算：
'''
print(train_data.shape)
print(train_targets)

'''
将神经网络输入所有采用不同范围的值都是有问题的。 网络可能能够
自动适应这种异构数据，但肯定会使学习变得更加困难。 处理此类数据
的一种广泛的最佳实践是进行特征标准化：对于输入数据中的每个特征（
输入数据矩阵中的一列），我们将减去特征的均值并除以标准差，因此
该特征将以0为中心，并具有单位标准偏差。
'''
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


'''
由于可用的样本很少，我们将使用一个非常小的网络，其中有两个隐藏层，
每个层有64个单元。 通常，拥有的训练数据越少，过度拟合就越差，
使用小型网络是缓解过度拟合的一种方法。
'''
# 搭建网络
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu',
                    input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # 验证数据预处理：通过k来划分
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 准备训练数据
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    # 创建已编译好的keras模型
    model = build_model()
    # 训练模型
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # 通过验证集验证模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)

K.clear_session()

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
