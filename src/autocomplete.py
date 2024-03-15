import random
from collections import defaultdict, Counter
from typing import List
import pandas as pd
import numpy as np

import nltk

from src.utils.utils import read_file

nltk.download('punkt')


def get_tokenized_data(data: str) -> List[List[str]]:
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


def replace_oov_words_by_unk(tokenized_sentences: List[List[str]], vocabulary: List[str], unknown_token="<unk>") -> \
        List[List[str]]:
    """
        Replace words not in the given vocabulary with '<unk>' token.
    :param tokenized_sentences: List of lists of strings
    :param vocabulary: List of strings that we will use
    :param unknown_token: A string representing unknown (out-of-vocabulary) words
    :return:
        - List of lists of strings, with words not in the vocabulary replaced
    """

    vocabulary_set = set(vocabulary)
    replaced_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = [token if token in vocabulary_set else unknown_token for token in sentence]
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences


def preprocess_data(train_data: List[List[str]], test_data: List[List[str]], count_threshold: int,
                    unknown_token: str = "<unk>") -> tuple[List[List[str]], List[List[str]], List[str]]:
    """
        Preprocesses the train and test data by replacing less common words with '<unk>' token and building vocabulary.

    :param train_data: List of lists of strings representing tokenized sentences in the training data.
    :param test_data: List of lists of strings representing tokenized sentences in the testing data.
    :param count_threshold: Minimum number of occurrences for a word to be included in the vocabulary.
    :param unknown_token: Token to represent unknown words. Defaults to "<unk>".

    :return:
        - Tuple containing preprocessed train data, preprocessed test data, and vocabulary.
    """
    # Build Vocabulary
    word_counts = defaultdict(int)
    for sentence in train_data:
        for token in sentence:
            word_counts[token] += 1

    vocabulary = [word for word, count in word_counts.items() if count >= count_threshold]

    # Replace less common words with "<unk>" in train and test data
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token)

    return train_data_replaced, test_data_replaced, vocabulary


def count_n_grams(data: List[List[str]], n: int, start_token='<s>', end_token='<e>'):
    """

    :param data: The input data where each element is a list of words.
    :param n: The number of words in each n-gram.
    :param start_token: The start token to pad the beginning of each sentence.
    :param end_token: The end token to pad the end of each sentence.
    :return:
        - n_gram: A Counter object mapping a tuple of n-words to its frequency.
    """
    # Pad each sentence with start and end tokens
    padded_data = [[start_token] * n + sentence + [end_token] for sentence in data]

    # Generate n-grams and count their frequencies
    n_grams = Counter()
    for sentence in padded_data:
        for i in range(len(sentence) - n + 1):
            n_gram = tuple(sentence[i: i + n])
            n_grams[n_gram] += 1
    return n_grams


def estimate_probabilities(previous_n_gram: list, n_gram_counts: dict, n_plus1_gram_counts: dict, vocabulary: List[str],
                           end_token='<e>', unknown_token="<unk>", k=1.0):
    """
        Estimate the probability of a next word using the n-gram counts with k-smoothing.

    :param previous_n_gram: A sequence of words of length n.
    :param n_gram_counts: Dictionary of counts of n-grams.
    :param n_plus1_gram_counts: Dictionary of counts of (n+1)-grams.
    :param vocabulary: List of words in the vocabulary.
    :param end_token: Token representing the end of a sentence. Default is '<e>'.
    :param unknown_token: Token representing unknown (out-of-vocabulary) words. Default is '<unk>'.
    :param k: Smoothing parameter. Default is 1.0.
    :return:
        - The estimated probability.
    """
    vocabulary = vocabulary + [end_token, unknown_token]
    vocabulary_size = len(vocabulary)
    probabilities = {}

    for word in vocabulary:

        previous_n_gram = tuple(previous_n_gram)
        previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
        denominator = previous_n_gram_count + k * vocabulary_size

        n_plus1_gram = previous_n_gram + (word,)
        n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
        numerator = n_plus1_gram_count + k

        probabilities[word] = numerator / denominator

    return probabilities


def make_count_matrix(n_plus1_gram_counts: dict, vocabulary: List[str]):
    """
        Create a count matrix from (n+1)-gram counts and a vocabulary.

    :param n_plus1_gram_counts: Dictionary of counts of (n+1)-grams.
    :param vocabulary: List of words in the vocabulary.
    :return:
        A DataFrame representing the count matrix.
    """
    # Add '<e>' and '<unk>' to the vocabulary
    vocabulary += ["<e>", "<unk>"]

    # Obtain unique n-grams
    n_grams = {n_plus1_gram[:-1] for n_plus1_gram in n_plus1_gram_counts}
    n_grams = list(set(n_grams))

    # Mapping from n-gram to row
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
    # Mapping from next word to column
    col_index = {word: j for j, word in enumerate(vocabulary)}

    # Initialize count matrix
    nrow, ncol = len(n_grams), len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))

    # Fill count matrix with n-gram counts
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram, word = n_plus1_gram[:-1], n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i, j = row_index[n_gram], col_index[word]
        count_matrix[i, j] = count

    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

def make_probability_matrix(n_plus1_gram_counts, unique_words, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>',
                         end_token='<e>', k=1.0):
    """
    Calculate perplexity for a list of sentences

    Args:
        sentence: List of strings
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: Positive smoothing constant

    Returns:
        Perplexity score
    """
    n = len(list(n_gram_counts.keys())[0])
    sentence = [start_token] * n + sentence + [end_token]
    sentence = tuple(sentence)
    N = len(sentence)
    product_pi = 1.0
    for t in range(None, None):
        n_gram = None
        word = None
        probability = None
        product_pi *= None
    perplexity = (product_pi) ** (1 / N)
    return perplexity


def run():
    data = read_file("./data/en_US.twitter.txt")
    tokenized_data = get_tokenized_data(data)
    random.seed(87)
    random.shuffle(tokenized_data)

    train_size = int(len(tokenized_data) * 0.8)
    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]

    minimum_freq = 2
    train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data, minimum_freq)


    sentences = [['i', 'like', 'a', 'cat'],
                     ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))
    bigram_counts = count_n_grams(sentences, 2)

    print('bigram counts')
    print('\ntrigram counts')
    trigram_counts = count_n_grams(sentences, 3)
    print(make_count_matrix(trigram_counts, unique_words))
    sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))
    bigram_counts = count_n_grams(sentences, 2)
    print("bigram probabilities")
    print(make_probability_matrix(bigram_counts, unique_words, k=1))
