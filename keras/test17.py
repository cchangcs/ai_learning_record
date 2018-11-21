import numpy as np

# 初始数据; 每个“样本”一个条目
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 首先，构建数据中所有token的索引
token_index = {}
for sample in samples:
    # 通过`split`方法对样本进行标记。实际使用时还会从样本中删除标点符号和特殊字符。
    for word in sample.split():
        if word not in token_index:
            # 为每个唯一单词指定唯一索引
            # 不将索引0赋值给任何单词
            token_index[word] = len(token_index) + 1

# 接下来，对样本进行矢量化
# 只考虑每个样本中的第一个'max_length'字
max_length = 10

# 用于存储结果
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1
print('==================================方式1=================================')
print(results)
# 字符级别的one-hot编码
import string

samples = ['The cat sat on the mat.', 'The dog ate my homeword.']
characters = string.printable  # 所有可打印的ASCII字符
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample[: max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1
print('==================================方式2=================================')
print(results)
# Keras具有内置实用程序，用于从原始文本数据开始在单词级别或字符级别执行单热编码文本。
# 这是实际使用的内容，因为它将处理许多重要的功能，例如从字符串中删除特殊字符，或者只接受数据集中的前N个最常用的单词（避免处理的常见限制） 非常大的输入向量空间）。

# 使用Keras进行字符级one-hot编码
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 创建一个tokenizer，配置为只考虑前1000个最常用的单词
tokenizer = Tokenizer(num_words=1000)

# 构建单词索引
tokenizer.fit_on_texts(samples)

# 可以直接获得一个热门的one-hot表示。
# 请注意，支持除one-hot编码之外的其他矢量化模式！
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# 恢复计算的单词索引的方法
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

'''
一热编码的变体是所谓的“one-hot hashing trick”，可以在词汇表中的唯一标记数量太大而无法明确处理时使用。不是明确地为每个单词分配索引并在字典
中保持这些索引的引用，而是可以将单词散列为固定大小的向量。这通常使用非常轻量级的散列函数来完成。
这种方法的主要优点是它不需要维护一个明确的单词索引，这可以节省内存并允许数据的在线编码（在看到所有可用数据之前立即开始生成令牌向量）。
这种方法的一个缺点是容易受到“哈希冲突”的影响：两个不同的词可能会以相同的哈希结束，随后任何查看这些哈希的机器学习模型都无法区分这些词之间的区别。
当散列空间的维度远大于被散列的唯一令牌的总数时，散列冲突的可能性降低。
'''
# 带散列技巧的one-hot编码
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# 如果有接近1000个单词（或更多），您将开始看到许多哈希冲突，这将降低此编码方法的准确性。
# 维度 = 1000
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[: max_length]:
        # 将单词哈希到一个介于0和1000之间的“随机”整数索引
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.
print('==================================方式3=================================')
print(results)