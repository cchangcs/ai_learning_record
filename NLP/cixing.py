import jieba.posseg as psg

s = '武汉大学是一所还不错的大学'
print([(x.word, x.flag) for x in psg.cut(s)])

print([(x.word, x.flag) for x in psg.cut(s) if x.flag.startswith('n')])