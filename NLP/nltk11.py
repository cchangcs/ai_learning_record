import nltk
from nltk.corpus import brown


def pos_feature_use_hist(sentence, i , history):
    features = {
        'suffix-1': sentence[i][-1:],
        'suffix-2': sentence[i][-2:],
        'suffix-3': sentence[i][-3:],
        'pre-word': 'START',
        'prev-tag': 'START'
    }
    if i > 0:
        features['pre-word'] = sentence[i - 1]
        features['prev-tag'] = history[i - 1]
    return features


class ContextPosTagger(nltk.TaggerI):
    def __init__(self, train):
        train_set = []
        for tagged_sent in train:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                features = pos_feature_use_hist(untagged_sent, i, history)
                train_set.append((features, tag))
                history.append(tag)
        print(train_set[:10])
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sent):
        history = []
        for i, word in enumerate(sent):
            features = pos_feature_use_hist(sent, i, history)
            tag = self.classifier.classify(features)
            history.append(tag)
        return zip(sent, history)

tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.8)
train_sets, test_sets = tagged_sents[0: size], tagged_sents[size:]

tagger = ContextPosTagger(train_sets)
tagger.classifier.show_most_informative_features()

print(tagger.evaluate(test_sets))
