import nltk
from nltk.corpus import brown

pattern = [
    (r'.*ing$','VBG'),
    (r'.*ed$','VBD'),
    (r'.*es$','VBZ'),
    (r'.*\'s$','NN$'),
    (r'.*s$','NNS'),
    (r'.*', 'NN')  #未匹配的仍标注为NN
]

brown_tagged_sents = brown.tagged_sents(categories='news')
train_num = int(len(brown_tagged_sents) * 0.9)

x_train = brown_tagged_sents[0: train_num]
x_test = brown_tagged_sents[train_num:]
t0 = nltk.RegexpTagger(pattern)
t1 = nltk.UnigramTagger(x_train, backoff=t0)
t2 = nltk.BigramTagger(x_train, backoff=t1)
print(t2.evaluate(x_test))
