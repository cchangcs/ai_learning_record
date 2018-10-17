from collections import Counter
import jieba

# 开启并行分词模式，参数为并发执行的进程数
liangjian_text = open('./liangjian.TXT', 'rb').read()
print(len(liangjian_text))
liangjian_words = [x for x in jieba.cut(liangjian_text) if len(x) >= 2]
c = Counter(liangjian_words).most_common(20)
print(c)

