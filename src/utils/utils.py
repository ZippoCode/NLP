import os
from typing import List


def read_file(filename: str) -> str:
    """
            Read lines from a file and return a list of strings.
    :param filename: Path to the file to read.
    :return:
        - data: A string representing the corpus of file.
    :raise FileNotFoundError: If the specified file does not exist.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    with open(filename, 'r') as f:
        data = f.read()

    return data


def read_lines(filename: str) -> List[str]:
    """
            Read lines from a file and return a list of strings.
    :param filename: Path to the file to read.
    :return:
        - List[str]: List of strings representing lines from the file.
    :raise FileNotFoundError: If the specified file does not exist.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    with open(filename, 'r') as f:
        data = f.readlines()

    return data


def read_vocabulary(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        vocabulary = f.read().split('\n')
    return vocabulary
