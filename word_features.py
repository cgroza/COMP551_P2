# Here we produce a list of words in our data that is also in the negative/positive lexicon
from data_loader import *

def get_training_words():
    training_data = LoadTrainingData("train/pos", "train/neg")

    training_words = set()
    for review in training_data.data:
        for word in review["review"]:
            training_words.add(word)

    with open("opinion-lexicon-English/positive-words.txt") as pos_f:
        positive_words = set(pos_f.read().split())

    with open("opinion-lexicon-English/negative-words.txt") as neg_f:
        negative_words = set(neg_f.read().split())

    # collect all positive words in our training set
    positive_train_words = []
    for train_word in training_words:
        if train_word in positive_words:
            positive_train_words.append(train_word)
    # write to file
    with open("positive_train_words.data", "w") as f:
        f.write(str(positive_train_words))

    # collect all negative words in our training set
    negative_train_words = []
    for train_word in training_words:
        if train_word in negative_words:
            negative_train_words.append(train_word)
    # write to file
    with open("negative_train_words.data", "w") as f:
        f.write(str(negative_train_words))

    return (positive_train_words, negative_train_words)
