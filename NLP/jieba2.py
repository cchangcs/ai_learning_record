import jieba
s = '武汉大学是一所还不错的大学'
cut = jieba.cut(s, cut_all=True)
print(','.join(cut))