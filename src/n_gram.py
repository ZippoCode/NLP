import nltk

nltk.download('punkt')


def sentence_to_n_gram(tokenized_sentence: list, n=3) -> list:
    """
        Generate n-grams from a tokenized sentence.

    :param tokenized_sentence: The list of words.
    :param n: The size of the n-grams. Defaults to 3.
    :return:
        - n_grams: A list of n-grams.
    """
    n_grams = [tokenized_sentence[i:i + n] for i in range(len(tokenized_sentence) - n + 1)]
    return n_grams


def main():
    sentence = 'i am happy because i am learning.'
    tokenized_sentence = nltk.word_tokenize(sentence)
    print(f'{sentence} -> {tokenized_sentence}')
    sentence_to_n_gram(tokenized_sentence, n=15)


if __name__ == '__main__':
    main()
