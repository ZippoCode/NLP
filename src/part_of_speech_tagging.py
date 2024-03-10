import string
from collections import defaultdict

import numpy as np

from src.utils.utils import read_lines, read_vocabulary

# Punctuation characters
punct = set(string.punctuation)
# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling",
               "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]


def assign_unk(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"


def preprocess(vocab, data_fp):
    """
    Preprocess data
    """
    orig, prep = [], []

    with open(data_fp, "r") as data_file:

        for cnt, word in enumerate(data_file):

            # End of sentence
            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue

            # Handle unknown words
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unk(word)
                prep.append(word)
                continue

            else:
                orig.append(word.strip())
                prep.append(word.strip())

    assert (len(orig) == len(open(data_fp, "r").readlines()))
    assert (len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep


def get_word_tag(line: str, vocab: set):
    """
        Extracts the (word, tag) pair from a line of the corpus.

    :param line: A line from the corpus containing a (word, tag) pair.
    :param vocab: Set of words in the vocabulary.
    :return:
        - tuple: A (word, tag) pair extracted from the line.
    """
    if not line.split():
        word = "--n--"
        tag = "--s--"
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
    return word, tag


def create_dictionaries(training_corpus: list, vocab: dict, verbose=True) -> tuple[dict, dict, dict]:
    """
        Create dictionaries for emission counts, transition counts, and tag counts based on the training corpus.

    :param training_corpus: A corpus where each line has a word followed by its tag.
    :param vocab: A dictionary where keys are words in the vocabulary, and values are indices.
    :param verbose: If True, print progress updates.

    :return:
        - emission_counts: A dictionary where the keys are (tag, word) and the values are the counts
        - transition_counts: A dictionary where the keys are (prev_tag, tag) and the values are the counts
        - tag_counts: A dictionary where the keys are the tags and the values are the counts
    """

    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    prev_tag = '--s--'
    for i, word_tag in enumerate(training_corpus, 1):
        word, tag = get_word_tag(word_tag, vocab)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1
        prev_tag = tag

        if i % 50000 == 0 and verbose:
            print(f"word count = {i}")

    return emission_counts, transition_counts, tag_counts


def predict_pos(prep: list, y: list, emission_counts: dict, vocab: dict, states: list) -> float:
    """

    :param prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
    :param y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
    :param emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
    :param vocab: a dictionary where keys are words in vocabulary and value is an index
    :param states:a sorted list of all possible tags for this assignment

    :return:
     - accuracy: Number of times you classified a word correctly
    """

    num_correct = 0
    total = 0

    for word, y_tup in zip(prep, y):
        y_tup_l = y_tup.split()
        if len(y_tup_l) == 2:
            true_label = y_tup_l[1]
        else:
            continue

        count_final = 0
        pos_final = ''
        if word in vocab:
            for pos in states:
                key = (pos, word)
                if key in emission_counts:
                    count = emission_counts[key]
                    if count > count_final:
                        count_final = count
                        pos_final = pos
            if pos_final == true_label:
                num_correct += 1
        total += 1
    accuracy = num_correct / total if total > 0 else 0

    return accuracy


def create_transition_matrix(alpha: float, tag_counts: dict, transition_counts: dict):
    """

    :param alpha: number used for smoothing
    :param tag_counts: a dictionary mapping each tag to its respective count
    :param transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
    :return:
         - transition_matrix: matrix of dimension (num_tags,num_tags)
    """
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)

    transition_matrix = np.zeros((num_tags, num_tags))

    for i, prev_tag in enumerate(all_tags):
        for j, current_tag in enumerate(all_tags):
            key = (prev_tag, current_tag)
            count = transition_counts.get(key, 0)
            count_prev_tag = tag_counts[prev_tag]

            transition_matrix[i, j] = (count + alpha) / (count_prev_tag + alpha * num_tags)

    return transition_matrix


def create_emission_matrix(alpha: float, tag_counts: dict, emission_counts: dict, vocab: dict):
    """

    :param alpha: tuning parameter used in smoothing
    :param tag_counts: a dictionary mapping each tag to its respective count
    :param emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
    :param vocab: a dictionary where keys are words in vocabulary and value is an index. Within the function it'll
                    be treated as a list
    :return:
     - emission_matrix: a matrix of dimension (num_tags, len(vocab))

    """
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab)
    emission_matrix = np.zeros((num_tags, num_words))

    for i, tag in enumerate(all_tags):
        for j, word in enumerate(vocab):
            count = emission_counts.get((tag, word), 0)
            count_tag = tag_counts[tag]

            emission_matrix[i, j] = (count + alpha) / (count_tag + alpha * num_words)
    return emission_matrix


def viterbi_forward(states: list, transition_matrix: np.ndarray, emission_matrix: np.ndarray, corpus: list, vocab: dict,
                    num_tags: int, verbose=True) -> tuple[np.ndarray, np.ndarray]:
    """

    :param states: a list of all possible parts-of-speech
    :param transition_matrix: Transition Matrix of dimension (num_tags, num_tags)
    :param emission_matrix: Emission Matrix of dimension (num_tags, len(vocab))
    :param corpus: a sequence of words whose POS is to be identified in a list
    :param vocab: a dictionary where keys are words in vocabulary and value is an index
    :param num_tags: the size of a dictionary mapping each tag to its respective count
    :param verbose: If "true" show in console output the state of the algorithm

    :return:
     - best_probs: matrix of dimension (num_tags, len(corpus)) of floats
     - best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    """
    num_words = len(corpus)

    best_probs = np.zeros((num_tags, num_words))
    best_paths = np.zeros((num_tags, num_words), dtype=int)

    s_idx = states.index("--s--")

    for i in range(num_tags):  # Initialization for the first word
        best_probs[i, 0] = np.log(transition_matrix[s_idx, i]) + np.log(emission_matrix[i, vocab[corpus[0]]])

    # Viterbi algorithm
    for i in range(1, num_words):
        if i % 5000 == 0 and verbose:
            print("Words processed: {:>8}".format(i))

        for j in range(num_tags):
            transition_probs = best_probs[:, i - 1] + np.log(transition_matrix[:, j])
            best_path_i = np.argmax(transition_probs)
            best_probs[j, i] = transition_probs[best_path_i] + np.log(emission_matrix[j, vocab[corpus[i]]])
            best_paths[j, i] = best_path_i

    return best_probs, best_paths


def viterbi_backward(best_probs: np.ndarray, best_paths: np.ndarray, corpus: list, states: list) -> list:
    """
        Perform Viterbi backward pass to find the most likely sequence of POS tags.

    :param best_probs: Matrix of the best probabilities obtained from the Viterbi forward pass.
    :param best_paths: Matrix of the best paths obtained from the Viterbi forward pass.
    :param corpus: A sequence of words.
    :param states: A list of all possible parts-of-speech.
    :return:
     - pred: The most likely sequence of POS tags for the given corpus.
    """
    m = best_paths.shape[1]
    z = [None] * m
    pred = [None] * m

    z[m - 1] = np.argmax(best_probs[:, m - 1])
    pred[m - 1] = states[z[m - 1]]

    for i in range(m - 1, 0, -1):
        pos_tag_for_word_i = z[i]
        z[i - 1] = best_paths[pos_tag_for_word_i, i]
        pred[i - 1] = states[z[i - 1]]

    return pred


def compute_accuracy(pred: list, y: list) -> float:
    """
        Compute the accuracy of the predictions

    :param pred: A list of predicted parts-of-speech.
    :param y: A list of lines where each word is separated by a '\t' (i.e., word \t tag).

    :return:
        - accuracy: The accuracy of the predictions compared to the true parts-of-speech.
    """
    num_correct = 0
    total = 0

    for prediction, y in zip(pred, y):
        word_tag_tuple = y.split()
        if len(word_tag_tuple) != 2:
            continue

        word, tag = word_tag_tuple
        num_correct += 1 if tag == prediction else 0
        total += 1

    return num_correct / total if total > 0 else 0


def main():
    # Read training corpus and vocabulary
    training_corpus = read_lines("../data/WSJ_02-21.pos")
    voc_l = read_vocabulary("../data/hmm_vocab.txt")

    # Get the index of the corresponding words.
    vocab = {word: i for i, word in enumerate(sorted(voc_l))}

    # Read test data and preprocess it
    y = read_lines("../data/WSJ_24.pos")
    _, prep = preprocess(vocab, "./data/test.words")

    # Create dictionaries and states
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
    states = sorted(tag_counts.keys())

    # Evaluate accuracy using predict_pos
    accuracy_predict_pos = predict_pos(prep, y, emission_counts, vocab, states)
    print(f"Accuracy of prediction using predict_pos: {accuracy_predict_pos:.4f}")

    # Configure smoothing parameter
    alpha = 0.001

    # Create transition and emission matrices
    A = create_transition_matrix(alpha, tag_counts, transition_counts)
    B = create_emission_matrix(alpha, tag_counts, emission_counts, vocab)

    # Run Viterbi algorithm
    best_probs, best_paths = viterbi_forward(states, A, B, prep, vocab, len(tag_counts))
    pred = viterbi_backward(best_probs, best_paths, prep, states)
    print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")


if __name__ == '__main__':
    main()
