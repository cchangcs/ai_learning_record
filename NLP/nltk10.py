import random
import nltk
from nltk.corpus import movie_reviews

docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()   # 类别
        for fileid in movie_reviews.fileids(category)]  # 文件标识符

# 将数据随机打乱
random.shuffle(docs)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
most_common_word = [word for (word, _) in all_words.most_common(2000)]


def doc_feature(doc):
    doc_words = set(doc)
    feature = {}
    for word in most_common_word:
        feature[word] = (word in doc_words)
    return feature

train_set = nltk.apply_features(doc_feature, docs[:100])
test_set = nltk.apply_features(doc_feature, docs[100:])

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features()

