
# coding: utf-8

# In[2]:


'''
使用convnet进行序列处理：
在Keras中，通过`Conv1D`层使用1D convnet，它具有与`Conv2D`非常相似的接口。
它需要具有shape`（样本，时间，特征）的3D张量输入，并且还返回类似形状的3D张量。
卷积窗口是时间轴上的1D窗口，输入张量中的轴1。
构建一个简单的2层1D convnet，并将其应用于IMDB情感分类任务。
这是获取和预处理数据的代码
'''
from  keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000  # 作为特征的单词数量
maxlen = 500  # 之后的文本全部截断

print('Loading data ...')


# In[5]:


(x_train, y_train), (x_test, y_test) = imdb .load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


# In[8]:


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)


# In[11]:


'''
1D convnets的结构：它们由一堆`Conv1D`和`MaxPooling1D`层组成，
最终以全局池层或`Flatten`结尾。 图层，将3D输出转换为2D输出，
允许将一个或多个“Dense”层添加到模型中，以进行分类或回归。
但是，一个不同之处在于我们可以负担得起使用带有1D网络的更大卷积窗口。 
实际上，对于2D卷积层，3×3卷积窗口包含3 * 3 = 9个特征向量，但是对于
1D卷积层，大小为3的卷积窗口将仅包含3个特征向量。 因此，
可以轻松地提供尺寸为7或9的1D卷积窗口。这是IMDB数据集的示例1D convnet：
'''
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()


# In[15]:


model.compile(optimizer=RMSprop(lr=1e-4),
             loss='binary_crossentropy',
             metrics=['acc'])
history = model.fit(x_train, y_train, 
         epochs=10,
         batch_size=128,
         validation_split=0.2)


# In[ ]:


'''
以下是训练和验证结果：验证准确性略低于LSTM，但运行时间更快，无论是在CPU还是GPU上。
重新训练此模型，并在测试集上运行它。 这是一个令人信服的证明，一维信号传输可以在字
级情绪分类任务上为循环网络提供快速，廉价的替代方案。
'''
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[20]:


'''
结合CNN和RNN来处理长序列由于1D convnets独立处理input patches，
因此它们对时间步长的顺序（超出局部尺度，卷积窗口的大小）不敏感，
与RNN不同。 当然，为了能够识别长期模式，可以堆叠许多卷积层和汇集层，
从而导致上层“看到”原始输入的长块 - 但这仍然是一个相当弱的方式 诱导顺序敏感性。
证明这一弱点的一种方法是温度预测问题的1D轮询，其中顺序敏感性是产生
良好预测的关键。 
'''
import os
import numpy as np

data_dir = 'jena_climate_2009_2016.csv'

f = open(data_dir)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

mean = float_data[: 200000].mean(axis=0)
float_data -= mean
std = float_data[: 200000].std(axis=0)
float_data /=  std

def generator(data, lookback, delay, min_index, max_index, 
             shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
            min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows), 
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows), ))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
    

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=200000,
                     shuffle=True,
                     step=step,
                     batch_size=batch_size)
val_gen = generator(float_data,
                   lookback=lookback,
                   min_index=200001,
                   max_index=300000,
                   step=step,
                   batch_size=batch_size)
test_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=300001,
                    max_index=None,
                    step=step,
                    batch_size=batch_size)

# 从val_gen`中抽取多少步骤以查看整个验证集：
val_steps = (300000 - 200001 - lookback) // batch_size

# 从‘test_val’中抽取多少以查看整个测试集
test_steps = (len(float_data) - 300001 - lookback) // batch_size


# In[ ]:


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                       input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.MaxPool1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                             steps_per_epoch=500,
                             epochs=5,  # 推荐训练20轮左右
                             validation_data=val_gen,
                             validation_steps=val_steps)


# In[ ]:


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


step = 3
lookback = 720  # Unchanged
delay = 144 # Unchanged

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step)
val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128


# In[ ]:



model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

