
# coding: utf-8

# In[1]:


# 下载语料库并将其转化为小写

import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))


# In[11]:


'''
接下来，将提取长度为“maxlen”的部分重叠序列，对它们进行one-hot
编码并将它们打包成形状为“（sequence，maxlen，unique_characters）”
的3D Numpy数组`x`。 
同时，准备一个包含相应目标的数组`y`：在每个提取序列之后的one-hot编码字符。
'''
# 提取的字符序列的长度
maxlen = 60

# 对每‘step’个字符序列采样一个新序列
step = 3

# 用于保存提取到的序列
sentences = []

# 用于保存targets
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(setences))

# 语料库中的唯一字符列表
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))

# 将唯一字符映射到`chars`中索引的字典
char_indices = dict((char, chars.index(char)) for char in chars)

# 接下来，将字符one-hot编码为二维数组
print('Vectorization...')
x = np.zeros((len(sentences),maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[15]:


'''
构建网络
网络是一个单独的'LSTM`层，后跟一个'Dense'分类器和所有可能字符的softmax。 
循环神经网络不是生成序列数据的唯一方法; 1D convnets也被证明非常成功。
'''
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))


# In[16]:


# 由于targets是one-hot编码，因此使用`categorical_crossentropy`作为训练模型的损失
optimizer = RMSprop(lr=1e-2)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

'''
训练语言模型并从中抽样
给定已训练的模型和原文本片段，重复生成新文本：
    * 1）从模型中得出目前可用文本的下一个字符的概率分布
    * 2）将分布重新调整到某个“temperature”
    * 3）根据重新加权的分布随机抽样下一个字符
    * 4）在可用文本的末尾添加新字符
  这是用来重新加权模型中出现的原始概率分布的代码，并从中绘制一个字符索引（“抽样函数”）：
'''
def sample(preds, temperatue=1.0):
    preds = np.array(preds).astype('float64')
    preds = np.log(preds) / temperatue
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[ ]:


'''
最后，反复训练和生成文本的循环。 开始在每个epoch之后使用一系列不同
的温度生成文本。 可以看到生成的文本在模型开始收敛时如何演变，
以及温度对抽样策略的影响。
'''
import random
import sys

for epoch in range(1, 60):
    print('epoch', epoch)
    # 在可用的训练数据上使模型适合1个epoch
    model.fit(x, y,
             batch_size=128,
             epochs=1)
    
    # 随机选择一个原文本片段
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed:“ ' + generated_text + ' ”')
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('-----temperature:', temperature)
        sys.stdout.write(generated_text)
        
        # 生成400字符
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1
            
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds=preds, temperatue=temperature)
            next_char = char[next_index]
            
            generated_text += next_char
            generated_text = generated_text[1:]
            
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# In[ ]:


'''
如上所示，低的temperature会产生极其重复且可预测的文本，但是在本地结构非常逼真的情况下：
特别是所有单词（一个单词是本地字符模式）都是真正的英语单词。随着温度的升高，生成的文本变得更有趣，令人惊讶，甚至创造性;它有时可能会发明一些听起来有些合理的新词（例如“eterned”或“troveration”）。在高温下，局部结构开始分解，大多数单词看起来像半随机字符串。毫无疑问，这里的0.5是这个特定设置中文本生成最有趣的温度。始终尝试多种采样策略！学习结构和随机性之间的巧妙平衡是让生成有趣的原因。
 请注意，通过训练更大的模型，更长的时间，更多的数据，您可以获得生成的样本，这些样本看起来比我们的更连贯和更真实。但是，当然，除了随机机会之外，不要期望生成任何有意义的文本：我们所做的只是从统计模型中采样数据，其中字符来自哪些字符。语言是一种通信渠道，通信的内容与通信编码的消息的统计结构之间存在区别。为了证明这种区别，这里有一个思想实验：如果人类语言在压缩通信方面做得更好，就像我们的计算机对大多数数字通信做的那样？那么语言就没那么有意义，但它缺乏任何内在的统计结构，因此无法像我们一样学习语言模型。
 拿走
 *我们可以通过训练模型来生成离散序列数据，以预测给定前一个令牌的下一个令牌。
 *在文本的情况下，这种模型被称为“语言模型”，可以基于单词或字符。
*采样下一个标记需要在遵守模型判断的可能性和引入随机性之间取得平衡。
 *处理这个的一种方法是_softmax temperature_的概念。总是尝试不同的温度来找到“正确”的温度。
'''

