import nltk
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
train_num = int(len(brown_tagged_sents) * 0.9)
x_train = brown_tagged_sents[0: train_num]
x_test = brown_tagged_sents[train_num:]
tagger = nltk.UnigramTagger(train=x_train)
print(tagger.evaluate(x_test))
