import nltk
from nltk.corpus import brown

pattern = [
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*es$', 'VBZ'),
    (r'.*\'s$', 'NN$'),
    (r'.*s$', 'NNS'),
    (r'.*', 'NN'),  # 未标注的仍为NN
]
sents = 'I am going to Wuhan University.'

tagger = nltk.RegexpTagger(pattern)

print(tagger.tag(nltk.word_tokenize(sents)))

tagged_sents = brown.tagged_sents(categories='news')
print(tagger.evaluate(tagged_sents))
