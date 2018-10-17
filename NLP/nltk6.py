import nltk
from nltk.corpus import  brown
sents = open('nothing_gonna_change_my_love_for_you.txt').read()
fdist = nltk.FreqDist(brown.words(categories='news'))
common_word = fdist.most_common(100)

cfdist = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
table = dict((word, cfdist[word].max()) for (word, _) in common_word)

uni_tagger = nltk.UnigramTagger(model=table, backoff=nltk.DefaultTagger('NN'))
print(uni_tagger.tag(nltk.word_tokenize(sents)))
tagged_sents = brown.tagged_sents(categories='news')
print(uni_tagger.evaluate(tagged_sents))

