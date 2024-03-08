from typing import List


def read_lines(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        training_corpus = f.readlines()
    return training_corpus


def read_vocabulary(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        vocabulary = f.read().split('\n')
    return vocabulary
