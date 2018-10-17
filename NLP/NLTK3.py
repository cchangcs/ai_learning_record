import nltk
sent = 'I am going to Wuhan University now'
tokens = nltk.word_tokenize(sent)

taged_sent = nltk.pos_tag(tokens)
print(taged_sent)


