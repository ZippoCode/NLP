import random
from collections import defaultdict
from typing import List

import nltk

from src.utils.utils import read_file

nltk.download('punkt')


def get_tokenized_data(data: str) -> List[str]:
    """
        Split data by linebreak "\n"
    :param data: A string
    :return:
        - tokens:  List of lists of tokens
    """
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    return [nltk.word_tokenize(sentence.lower()) for sentence in sentences]


def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences

    Args:
        tokenized_sentences: List of lists of strings

    Returns:
        dict that maps word (str) to the frequency (int)
    """

    word_counts = defaultdict(int)
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] += 1

    return word_counts


def run():
    data = read_file("./data/en_US.twitter.txt")
    tokenized_data = get_tokenized_data(data)
    random.seed(87)
    random.shuffle(tokenized_data)

    train_size = int(len(tokenized_data) * 0.8)
    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]
    print("{} data are split into {} train and {} test set".format(
        len(tokenized_data), len(train_data), len(test_data)))

    print("First training sample:")
    print(train_data[0])

    print("First test sample")
    print(test_data[0])
