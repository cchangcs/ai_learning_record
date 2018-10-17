import jieba
s = '武汉大学是一所还不错的大学'
cut = jieba.cut_for_search(s)
print(','.join(cut))