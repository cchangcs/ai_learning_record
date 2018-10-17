import jieba
s = '武汉大学是一所还不错的大学'
result = jieba.cut(s)
print(','.join(result))
