import numpy as np
from collections import OrderedDict
from itertools import combinations

class CoocurrenceMatrix:
    def __init__(self, tokens: list[str], window_size: int = 3) -> None:
        self.tokens = tokens
        self.window_size = window_size
        self.vocabulary = OrderedDict()
    
    def build_vocabulary(self) -> None:
        """
        Build vocabulary in the order of first appearance of the word in the text.
        """
        self.vocabulary.clear()
        for token in self.tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
    
    def build_matrix(self): #-> tuple(np.ndarray(np.int32), list[str]):
        """
        Create n by n matrix of co-occurence counts in sliding window.
        """
        self.build_vocabulary()
        vocabulary_size = len(self.vocabulary)

        # Initialize matrix
        matrix = np.zeros((vocabulary_size, vocabulary_size), dtype = np.int32)

        # Count co-occurences
        for i in range(len(self.tokens) - self.window_size + 1):
            window = self.tokens[i : (i + self.window_size)]

            for word1, word2 in combinations(window, 2):
                index1 = self.vocabulary[word1]
                index2 = self.vocabulary[word2]
                matrix[index1, index2] += 1
                matrix[index2, index1] += 1

        vocabulary_list = list(self.vocabulary.keys())
        return matrix, vocabulary_list
