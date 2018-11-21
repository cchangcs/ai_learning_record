from keras.datasets import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    # 创建一个全0矩阵->shape(len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# 将训练数据和测试数据矢量化
X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

# 矢量化标签
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

'''
应对overfiting：
    减少网络规模
    防止过拟合的最简单的方法是减少模型的大小，即减少模型中参数的数量（通常
    被称为模型的“容量”，具有更多参数的模型将具有更好的“学习能力”，因此将能
    够容易地学习训练样本与其目标之间的完美的字典式映射，没有任何泛化能力的映射。
    例如，可以很容易地建立具有500,000个二进制参数的模型来学习MNIST训练集中的每个
    数字的类：我们对于50,000个数字中的每一个只需要10个二进制参数。这样的模型对于分
    类新的数字样本是没有用的。始终牢记这一点：深度学习模型往往善于适应训练数据，
    但真正的挑战是概括，而不是适合。
    另一方面，如果网络具有有限的记忆资源，它将不能容易地学习这种映射，因此，为了使其
    损失最小化，它将不得不求助于学习具有关于该预测能力的预测能力的压缩表示。目标 -
    正是我们感兴趣的表示类型。同时，请记住，您应该使用具有足够参数的模型，使其不适合：
    您的模型不应该缺乏记忆资源。在“太多容量”和“容量不足”之间存在折衷。

    不幸的是，没有神奇的公式来确定正确的层数是多少，或者每层的正确尺寸是多少。您必须评
    估一组不同的体系结构（在您的alidation集上，当然不在您的测试集上），以便为您的数
    据找到合适的模型大小。找到合适的模型大小的一般工作流程是从相对较少的图层和参数开始，
    然后开始增加图层的大小或添加新图层，直到您看到有关验证损失的收益递减。
'''

original_model = Sequential()
original_model.add(Dense(16, activation='relu', input_shape=(10000, )))
original_model.add(Dense(16, activation='relu'))
original_model.add(Dense(1, activation='sigmoid'))

original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])

# smaller network
# 较小的网络开始过度拟合，而不是参考网络（在6个时期之后而不是4个时期），并且一旦开始过度拟合，其性能降低得慢得多。
smaller_model = Sequential()
smaller_model.add(Dense(4, activation='relu', input_shape=(10000,)))
smaller_model.add(Dense(4, activation='relu'))
smaller_model.add(Dense(1, activation='sigmoid'))

smaller_model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

# 以下是原始网络和较小网络的验证损失的比较,点是较小网络的验证损失值，十字架是初始网络（请记住：较低的验证损失表示更好的模型
original_hist = original_model.fit(X_train, y_train,
                                   epochs=20,
                                   batch_size=512,
                                   validation_data=(X_test, y_test))

smaller_model_fit = smaller_model.fit(X_train, y_train,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(X_test, y_test))

epochs = range(1, 21)
original_val_loss = original_hist.history['val_loss']
smaller_model_fit_loss = smaller_model_fit.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, smaller_model_fit_loss, 'bo', label='Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()
# 在一个迭代之后，更大的网络几乎立即开始过度拟合，并且更严重地装配。 它的验证损失也更加嘈杂。
bigger_model = Sequential()
bigger_model.add(Dense(512, activation='relu', input_shape=(10000,)))
bigger_model.add(Dense(512, activation='relu'))
bigger_model.add(Dense(1, activation='sigmoid'))

bigger_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['acc'])

bigger_model_hist = bigger_model.fit(X_train, y_train,
                                     epochs=20,
                                     batch_size=512,
                                     validation_data=(X_test, y_test))

bigger_model_val_loss = bigger_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_val_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.show()

'''
较大的网络很快就会使其训练损失接近于零。网络容量越大，能够越快地对训练数据进行建模（导致训练损失低），
但过度拟合的可能性越大（导致训练和验证损失之间的差异很大）。

##添加权重正则化
您可能熟悉_Occam的Razor_原则：给出两个解释的东西，最可能正确的解释是“最简单”的解释，即做出最少
量假设的解释。这也适用于神经网络学习的模型：给定一些训练数据和网络架构，有多组权重值（多个_models_）
可以解释数据，而简单模型比复杂模型更不容易过度拟合。

在这种情况下，“简单模型”是一个模型，其中参数值的分布具有较少的熵（或者具有较少参数的模型，如我们在上
面的部分中所见）。因此，减轻过度拟合的常用方法是通过强制其权重仅采用较小的值来对网络的复杂性施加约束，
这使得权重值的分布更“规则”。这被称为“权重正则化”，并且通过向网络的损失函数添加与具有大权重相关联的
_cost_来完成。这个成本有两种：

* L1正则化，其中增加的成本与权重系数的_绝对值成比例_（即所谓的
权重的“L1规范”）。
* L2正则化，其中所添加的成本与权重系数的值的_square成比例_（即，称为权重的“L2范数”）。 L2正则化在
神经网络的背景下也称为_weight decay_。不要让不同的名字让你感到困惑：重量衰减在数学上与L2正则化完全相同。

在Keras中，通过将_weight regularrizer instances_作为关键字参数传递给图层来添加权重正则化。'''

# l2（0.001）`表示层的权重矩阵中的每个系数都会将“0.001 * weight_coefficient_value”
# 添加到网络的总损失中。 请注意，由于此惩罚仅在训练时间_添加，因此在训练时该网络的损失将远高于测试时间。
from keras import regularizers
l2_model = Sequential()
l2_model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001),
                   activation='relu', input_shape=(10000, )))
l2_model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001),
                   activation='relu'))
l2_model.add(Dense(1, activation='sigmoid'))

l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])

l2_model_hist = l2_model.fit(X_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(X_test, y_test))

l2_model_val_loss = l2_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()

'''
Dropout是由Hinton和他在多伦多大学的学生开发的最有效和最常用的神经网络正则化技术之一。应用于层的
丢失包括在训练期间随机“退出”（即设置为零）该层的多个输出特征。假设一个给定的层通常会在训练期间为给
定的输入样本返回一个矢量“[0.2,0.5,1.3,0.8,1.1]”;在应用了丢失之后，该向量将具有随机分布的几个
零条目，例如， `[0,0.5,1.3,0,1.1]`。 “Dropout”是被淘汰的特征的一部分;它通常设置在0.2和0.5之间。
在测试时，没有单位被剔除，而是将图层的输出值按比例缩小等于Dropout的因子，以便平衡更多单位活跃的事实而不是训练时间。
考虑一个Numpy矩阵，它包含一个层的输出，`layer_output`，形状`（batch_size，features）`。在训练时，我们会随机将矩阵中的一小部分值归零：
'''
dpt_model = Sequential()
dpt_model.add(Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(Dropout(0.5))
dpt_model.add(Dense(16, activation='relu'))
dpt_model.add(Dropout(0.5))
dpt_model.add(Dense(1, activation='sigmoid'))

dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

dpt_model_hist = dpt_model.fit(X_train, y_train,
                              epochs=20,
                              batch_size=512,
                              validation_data=(X_test, y_test))

dpt_model_val_loss = dpt_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()