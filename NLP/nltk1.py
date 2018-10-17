import nltk
import jieba

raw = open('liangjian.TXT', 'rb').read()
# jieba.lcut()与jieba.cut()的区别在于：
# jieba.cut()返回一个可迭代的generator，可以使用for循环获得粉刺后得到的每一个词语
# jieba.lcut()直接返回list
text = nltk.text.Text(jieba.lcut(raw))

# 显示出现的上下文
print(text.concordance(u'驻岛'))

# 对同义词的使用习惯，显示words出现的相同模式
print(text.common_contexts(['小日本', '鬼子']))

# 显示最常用的二次搭配
print(text.collocations())

# 查看关心的词在文中出现的位置
text.dispersion_plot(['李云龙', '秀芹'])
