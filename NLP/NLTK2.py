import nltk
text = open('nothing_gonna_change_my_love_for_you.txt').read()
fdist = nltk.FreqDist(nltk.word_tokenize(text))
# 显示累积频率分布图
fdist.plot(30, cumulative=True)