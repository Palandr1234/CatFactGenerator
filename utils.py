from collections import defaultdict
import numpy as np


def create_word_list(path):
    words = []

    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8', 'ignore')
            line = line.strip()
            words.append(line)
    return words


class WordDictionary(object):
    def __init__(self, path):
        words = create_word_list(path)
        self.voc_size = len(words)

        index_range = range(len(words))
        mapping = zip(words, index_range)
        self.d = defaultdict(lambda: 1, mapping)

    def __getitem__(self, item):
        return self.d[item]

    def __contains__(self, item):
        return item in self.d

    def __len__(self):
        return len(self.d)

    def inverse(self):
        return {v: k for k, v in self.d.items()}


class Dataset(object):
    def __init__(self, sentences, sizes):
        if not isinstance(sentences, list):
            sentences = [sentences]
            sizes = [sizes]
        self.sentences = sentences
        self.sizes = sizes
        self.num_items = sum(len(array) for array in sizes)
        self.next_batch_idx = 0
        self.last_matrix_idx = 0
        self.epoch_counter = 0
        self.largest_len = max(sentence.shape[1] for sentence in sentences)

    def __len__(self):
        return self.num_items

    def reset_epoch_counter(self):
        self.epoch_counter = 0

    def next_batch(self, batch_size):
        matrix = self.sentences[self.last_matrix_idx]
        if self.next_batch_idx >= len(matrix):
            self.last_matrix_idx += 1
            if self.last_matrix_idx >= len(self.sentences):
                self.epoch_counter += 1
                self.last_matrix_idx = 0
            self.next_batch_idx = 0
            matrix = self.sentences[self.last_matrix_idx]

        sizes = self.sizes[self.last_matrix_idx]
        from_idx = self.next_batch_idx
        to_idx = from_idx + batch_size
        batch_sentences = matrix[from_idx: to_idx]
        batch_sizes = sizes[from_idx: to_idx]
        self.next_batch_idx = to_idx

        return batch_sentences, batch_sizes

    def join_matrices(self, eos, max_size=None, shuffle=True):
        if max_size is None:
            max_size = max(matrix.shape[1] for matrix in self.sentences)
        padded_matrices = []
        for matrix in self.sentences:
            if matrix.shape[1] == max_size:
                padded = matrix
            else:
                difference = max_size - matrix.shape[1]
                padded = np.pad(matrix, [(0, 0), (0, difference)], 'constant', constant_values=eos)
            padded_matrices.append(padded)
        sentences = np.vstack(padded_matrices)
        sizes = np.hstack(self.sizes)
        if shuffle:
            np.random.shuffle(sentences)
            np.random.shuffle(sizes)
        return sentences, sizes


