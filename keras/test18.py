# _*_ coding:utf-8 _*_
from keras.layers import Embedding

# 嵌入层至少需要两个参数：
# 可能的token数量，这里是1000（1 +最大单词索引），和嵌入的维度，这里64。
embedding_layer = Embedding(1000, 64)

'''
“Embedding”层最好被理解为将整数索引（代表特定单词）映射到密集向量的字典。
它将整数作为输入，将这些整数查找到内部字典中，并返回相关的向量。它实际上是一个字典查找。

“Embedding”层将整数的二维张量作为输入，形状为“（samples，sequence_length）”，
其中每个条目是一个整数序列。它可以嵌入可变长度的序列，所以例如我们可以在批次之上输入我们的嵌入层，
这些批次可以具有形状`（32,10）`（32个序列长度为10的批次）或`（64,15）`（批量） 64个长度的序列15）。
然而，批处理中的所有序列必须具有相同的长度（因为我们需要将它们打包成单个张量），因此比其他序列短的序列应
该用零填充，并且应该截断更长的序列。
该层返回一个形状为“（samples，sequence_length，embedding_dimensionality）”的3D浮点张量。
然后可以通过RNN层或1D卷积层处理这样的3D张量（两者将在下面的部分中介绍）。
 
当你实例化一个'Embedding`层时，它的权重（它的token向量的内部字典）最初是随机的，就像任何其他层一样。
在训练期间，这些单词向量将通过反向传播逐步调整，将空间构造成下游模型可以利用的东西。一旦完全训练，嵌入空间将显示
许多结构 - 一种专门针对训练模型的特定问题的结构。
让我们将这个想法应用到IMDB电影评论情绪预测任务中。准备数据。将电影评论限制在最常见的10,000个单词中，并在仅20个
单词后剪切评论。网络将简单地为10,000个单词中的每个单词学习8维嵌入，将输入整数序列（2D整数张量）转换为嵌入序列
（3D浮动张量），将张量平坦化为2D，并训练单个“密集”层最高分类。
'''
from keras.datasets import imdb
from keras import preprocessing

# 考虑作为特征的单词数
max_features = 10000
# 在max_features数量的单词后剪切文本
max_len = 20

# 将数据作为整数列表进行加载
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 这将我们的整数列表转换为shape：`（samples，maxlen）`的2D整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# 指定嵌入层的最大输入长度，以便稍后展平嵌入的输入
model.add(Embedding(10000, 8, input_length=max_len))

# 在嵌入层之后，激活形状为`（samples，maxlen，8）`。
# 将嵌入的3D张量展平为2D张量的形状`（samples，maxlen * 8）`
model.add(Flatten())

# 在顶部添加分类器
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=32,
                 validation_split=0.2)

'''
我们得到的验证准确率为~76％，考虑到我们只查看每个评论中的前20个单词，这是非常好的。但请注意，仅仅展平嵌入的序列并在顶部训练
单个“密集”层会导致模型分别处理输入序列中的每个单词，而不考虑词间关系和结构句子（例如，它可能会同时处理
两个_“这部电影是狗屎“_和_”这部电影是狗屎“_ as as negative”评论“）。在嵌入序列的顶部添加循环层或1D卷积层会更好，以学习将每
个序列作为一个整体考虑在内的特征。这就是我们将在接下来的几节中关注的内容。

使用预先训练的单词嵌入
有时候，您可以获得的培训数据非常少，无法单独使用您的数据来学习适当的任务特定的词汇表嵌入。该怎么办？
您可以从已知高度结构化的预先计算的嵌入空间中加载嵌入向量，并展示有用的属性，而不是学习嵌入与您想要解决的问题的嵌入向量，
从而捕获语言结构的通用方面。在自然语言处理中使用预先训练的单词嵌入背后的基本原理与在图像分类中使用预训练的网络非常相似：
我们没有足够的数据来学习我们自己的真正强大的功能，但我们期望这些功能我们需要相当通用，即常见的视觉特征或语义特征。在这种情况下，
重用在不同问题上学习的特征是有意义的。
这样的单词嵌入通常使用单词出现统计（关于在句子或文档中共同出现的单词的观察），使用各种技术来计算，一些涉及神经网络，另一些则不是。
Bengio等人最初探讨了以无人监督的方式计算的密集，低维度的文字嵌入空间的想法。在21世纪初期，它发布了最着名和最成功的一词嵌入方案之后才
开始真正起飞研究和行业应用：2013年由Mikolov在谷歌开发的Word2Vec算法.Word2Vec维度捕获特定的语义属性，例如性别。
有各种预先计算的字嵌入数据库，可以下载并开始在Keras“嵌入”层中使用。 Word2Vec就是其中之一。另一个流行的被称为“GloVe”，由斯坦福大学的
研究人员在2014年开发。它代表“用于词表示的全局向量”，它是一种基于对词共现统计矩阵进行因式分解的嵌入技术。它的开发人员为数以百万计的英语
令牌提供了预先计算的嵌入，从维基百科数据或公共爬网数据中获得。
让我们来看看如何开始在Keras模型中使用GloVe嵌入。当然，相同的方法对于Word2Vec嵌入或您可以下载的任何其他字嵌入数据库都是有效的。
我们还将使用此示例来刷新我们在几段前介绍的文本标记化技术：我们将从原始文本开始，然后继续前进。

把它们放在一起：从原始文本到文字嵌入
我们将使用类似于我们刚刚过去的模型 - 在矢量序列中嵌入句子，展平它们并在顶部训练“密集”层。但我们将使用预先训练的字嵌入来实现，而不是使用
Keras中打包的预标记化IMDB数据，我们将从头开始，通过下载原始文本数据。

以原始文本下载IMDB数据
首先，在 http://ai.stanford.edu/~amaas/data/sentiment/`下载原始IMDB数据集，解压缩它。
现在将各个训练评论收集到一个字符串列表中，每个评论一个字符串，然后还将评论标签（正面/负面）收集到“标签”列表中：
'''
import os

imdb_dir = 'aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# 标记数据
# 对我们收集的文本进行矢量化，并准备培训和验证分割。
# 因为预训练的单词嵌入对于几乎没有可用训练数据的问题特别有用（否则，特定于任务的嵌入可能胜过它们），
# 添加以下内容：将训练数据限制在前200样本。 因此，在查看了200个例子后，将学习如何对电影评论进行分类......

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

max_len = 100  # 将100个单词后的评论截断
training_samples = 200  # 训练200个样本
validation_samples = 10000  # 进队10000个样本进行验证
max_words = 10000  # 值考虑数据集中的前10000个词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens:' % len(word_index))

data = pad_sequences(sequences, maxlen=max_len)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将数据拆分为训练集和验证集。但首先，对数据进行清洗，从有序样本数据开始（首先是负数，然后是所有正数）。
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

'''
下载GloVe单词嵌入
前往`https://nlp.stanford.edu/projects/glove/`（您可以在其中了解有关GloVe算法的更多信息），
并从2014英语维基百科下载预先计算的嵌入。 这是一个名为`glove.6B.zip`的822MB zip文件，包含400,000个单词（或非单词标记）的100维嵌入向量。 解开它。

预处理嵌入
让我们解析未压缩的文件（它是一个`txt`文件）来构建一个索引，将单词（作为字符串）映射到它们的向量表示（作为数字向量）。
'''
glove_dir = 'glove'

embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

'''
构建一个嵌入矩阵，将它加载到“嵌入”层。 它必须是形状矩阵`（max_words，embedding_dim）`，
其中每个条目`i`在参考词索引（在标记化期间建立）中包含索引`i`的单词的'embedding_dim`维向量。
请注意，索引“0”不应代表任何单词或标记 - 它是占位符。
'''
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if i < max_words:
        # 嵌入索引中找不到的单词将全为零。
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 定义一个模型
# 将使用与以前相同的模型架构：
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

'''
将GloVe嵌入加载到模型中
“嵌入”层具有单个权重矩阵：2D浮点矩阵，其中每个条目“i”是意图与索引“i”相关联的单词向量。
 很简单。 将准备好的GloVe矩阵加载到'Embedding`层中，这是模型中的第一层：
'''
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = True

'''
另外，冻结嵌入层（将其“可训练”属性设置为“False”），遵循与预先训练的信号特征背景下相同的基本原理：当模型的某些部分预先存在时
训练（如'嵌入'层），部件随机初始化（如我们的分类器），训练前不应更新训练前的部分，以免忘记他们已经知道的内容。
 由随机初始化的层触发的大梯度更新对于已经学习的特征将是非常具有破坏性的。
'''
# 训练和评估
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=32,
                 validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

import matplotlib.pyplot as plt

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(acc) + 1)

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

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))


# In[22]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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


test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=max_len)
y_test = np.asarray(labels)



model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)