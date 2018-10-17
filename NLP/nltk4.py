import nltk
from nltk.corpus import brown

default_tagger = nltk.DefaultTagger('NN')
sents = 'I am going to Wuhan University now.'
print(default_tagger.tag(sents))

tagged_sent = brown.tagged_sents(categories='news')
print(default_tagger.evaluate(tagged_sent))
