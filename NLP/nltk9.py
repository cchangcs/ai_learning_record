import nltk
import json

lines = open('199801.txt', 'rb').readlines()
all_tagged_sents = []

for line in lines:
    line = line.decode('utf-8')
    sent = line.split()
    tagged_sent = []
    for item in sent:
        pair = nltk.str2tuple(item)
        tagged_sent.append(pair)

    if len(tagged_sent) > 0:
        all_tagged_sents.append(tagged_sent)

train_size = int(len(all_tagged_sents) * 0.8)
x_train = all_tagged_sents[: train_size]
x_test = all_tagged_sents[train_size:]
tagger = nltk.UnigramTagger(train=x_train, backoff=nltk.DefaultTagger('n'))

tokens = nltk.word_tokenize(u'我 认为 不丹 的 被动 卷入 不 构成 此次 对峙 的 主要 因素。')
tagged = tagger.tag(tokens)
print(json.dumps(tagged, ensure_ascii=False))
print(tagger.evaluate(x_test))

