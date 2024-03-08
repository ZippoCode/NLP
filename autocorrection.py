import re
import sys
from collections import defaultdict

import numpy as np
from tabulate import tabulate


def process_data(file_name: str):
    """

    :param file_name:
    :return words: a list containing all the words in the corpus in lower case.
    """
    with open(file_name) as file:
        text_lowercase = file.read().lower()
    words = re.findall(r'\w+', text_lowercase)

    return words


def get_count(words: list):
    """
    
    :param words: a list of words representing the corpus.
    :return: The wordcount dictionary where key is the word and value is its frequency.
    """
    word_count_dict = defaultdict(int)
    for word in words:
        word_count_dict[word] += 1

    return word_count_dict


def get_probs(word_count_dict: dict):
    """

    :param word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    :return: A dictionary where keys are the words and the values are the probability that a word will occur.
    """

    total_words = sum(word_count_dict.values())
    probs = {word: count / total_words for word, count in word_count_dict.items()}
    return probs


def delete_letter(word: str, verbose=False):
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    delete_l = [L + R[1:] for L, R in split_l if R]

    if verbose:
        print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")
    return delete_l


def switch_letter(word: str, verbose=False):
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    switch_l = [L[:-1] + R[0] + L[-1] + R[1:] for L, R in split_l if len(L) and len(R) > 0]

    if verbose:
        print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l


def replace_letter(word: str, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    replace_l = [L + char + R[1:] for char in letters for L, R in split_l if R if (L + char + R[1:]) != word]
    replace_set = set(replace_l)
    replace_l = sorted(list(replace_set))

    if verbose:
        print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l


def insert_letter(word, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    insert_l = [L + char + R for char in letters for L, R in split_l]

    if verbose:
        print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l


def edit_one_letter(word: str, allow_switches=True) -> set:
    """
    Generate a set of all possible strings that are one edit away from the input word.

    :param word: The input word.
    :param allow_switches: Whether to allow letter switches. Defaults to True.

    :return:
    - set: A set of strings with one edit distance from the input word.
    """
    edits = set()

    edits.update(delete_letter(word))
    if allow_switches:
        edits.update(switch_letter(word))
    edits.update(replace_letter(word))
    edits.update(insert_letter(word))

    return edits


def edit_two_letters(word, allow_switches=True) -> set:
    """
    Generate a set of all possible strings that are two edits away from the input word.

    :param word: The input word.
    :param allow_switches: Whether to allow letter switches. Defaults to True.

    :return:
    - set: A set of strings with two edit distances from the input word.
    """
    edits = edit_one_letter(word, allow_switches)
    edit_two_set = [edit_two for edit in edits for edit_two in edit_one_letter(edit, allow_switches)]
    sorted(edit_two_set)

    return set(edit_two_set)


def get_corrections(word: str, probs: dict, vocab: set, n=2, verbose=False) -> list:
    """

    :param word:
    :param probs:
    :param vocab:
    :param n:
    :param verbose:
    :return:
    """
    dict_word = word if word in vocab else []
    one_edits = [word for word in edit_one_letter(word)]
    two_edits = [word for word in edit_two_letters(word)]
    suggestions = dict_word or one_edits or two_edits

    best_words = {suggestion: probs[suggestion] if suggestion in probs else 0 for suggestion in suggestions}
    sorted_best_words = sorted(best_words.items(), key=lambda x: x[1], reverse=True)

    n_best = sorted_best_words[:n]

    if verbose:
        print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best


def min_edit_distance(source, target, insert_cost=1, delete_cost=1, replace_cost=2):
    """
    Compute the minimum edit distance matrix between source and target strings

    :param source: A string corresponding to the string you are starting with
    :param target: A string corresponding to the string you want to end with
    :param insert_cost: An integer setting the insert cost
    :param delete_cost: An integer setting the delete cost
    :param replace_cost: An integer setting the replacement cost
    :return:
        - distances: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        - med: the minimum edit distance (med) required to convert the source string to the target
    """
    m = len(source)
    n = len(target)
    distances = np.zeros((m + 1, n + 1), dtype=int)

    for row in range(1, m + 1):
        distances[row, 0] = distances[row - 1, 0] + delete_cost

    for col in range(1, n + 1):
        distances[0, col] = distances[0, col - 1] + insert_cost

    for row in range(1, m + 1):
        for col in range(1, n + 1):
            r_cost = 0 if source[row - 1] == target[col - 1] else replace_cost

            distances[row, col] = min(
                distances[row - 1, col] + delete_cost,
                distances[row, col - 1] + insert_cost,
                distances[row - 1, col - 1] + r_cost
            )

    med = distances[m, n]

    return distances, med


def min_edit_distance_with_backtrace(source: str, target: str, insert_cost=1, delete_cost=1, replace_cost=2):
    """
    Compute the minimum edit distance matrix including the backtrace between source and target strings

    
    :param source: A string corresponding to the string you are starting with
    :param target: A string corresponding to the string you want to end with
    :param insert_cost: An integer setting the insert cost
    :param delete_cost: An integer setting the delete cost
    :param replace_cost: An integer setting the replacement cost
    :return:
        - distances: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        - med: the minimum edit distance (med) required to convert the source string to the target
    """
    m = len(source)
    n = len(target)
    top_char, left_char, top_left_char = "↑ ", "← ", "↖ "

    distances = np.empty((m + 1, n + 1), dtype=tuple)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            distances[i, j] = ("", 0)

    for row in range(1, m + 1):  # Fill in column 0, from row 1 to row m, both inclusive
        distances[row, 0] = (top_char, distances[row - 1, 0][1] + delete_cost)

    for col in range(1, n + 1):  # Fill in row 0, for all columns from 1 to n, both inclusive
        distances[0, col] = (left_char, distances[0, col - 1][1] + insert_cost)

    for row in range(1, m + 1):
        for col in range(1, n + 1):
            r_cost = 0 if source[row - 1] == target[col - 1] else replace_cost
            delete = distances[row - 1, col][1] + delete_cost
            insert = distances[row, col - 1][1] + insert_cost
            replace = distances[row - 1, col - 1][1] + r_cost
            min_value = min(delete, insert, replace)

            # Backtrace
            direction = ""
            if min_value == delete:
                direction += top_char
            if min_value == insert:
                direction += left_char
            if min_value == replace:
                direction += top_left_char

            distances[row, col] = (direction, min_value)

    med = distances[m, n][1]

    return distances, med


def print_matrix_backtrace(matrix, source: str, target: str):
    headers = [letter for letter in "#" + target]
    source = "#" + source
    rows = []

    for i, row in enumerate(matrix):
        if len(row) == 2:
            row_values = [f"{str(cell[0])} {cell[1]}" for cell in row]
        else:
            row_values = [f"{cell}" for cell in row]
        rows.append([f"{source[i]}"] + row_values)

    print(tabulate(rows, headers, tablefmt="pretty"))


def main():
    source = 'intention'
    target = 'execution'
    matrix, min_edits = min_edit_distance(source, target)
    print("minimum edits: ", min_edits, "\n")
    print_matrix_backtrace(matrix, source, target)


if __name__ == '__main__':
    main()
    sys.exit()
