import nltk
gold = [1, 2, 3, 4]
test = [1, 3, 2, 4]
print(nltk.ConfusionMatrix(gold, test))