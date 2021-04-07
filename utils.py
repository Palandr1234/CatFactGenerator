from collections import defaultdict
import numpy as np


def create_line_list(path):
    # read the data from text file and return a list of lines
    # path - path to the text file
    words = []

    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8', 'ignore')
            # delete all whitespaces in the line
            line = line.strip()
            words.append(line)
    return words


class WordDictionary(object):
    # a class for word dictionary, vocabulary size and padding
    def __init__(self, path):
        words = create_line_list(path)
        self.voc_size = len(words)

        # Map the words to the appropriate indexes
        # If there were errors during reading the file, they will be represented as an empty string
        index_range = range(len(words))
        mapping = zip(words, index_range)
        self.unk_index = words.index('<unk>')
        self.d = defaultdict(lambda: self.unk_index, mapping)
        self.eos_index = self.d['</s>']

    def __getitem__(self, item):
        return self.d[item]

    def __contains__(self, item):
        return item in self.d

    def __len__(self):
        return len(self.d)

    def inverse(self):
        # Return a mapping from index to the word
        return {v: k for k, v in self.d.items()}


class Dataset(object):
    # class for the autoencoder dataset
    def __init__(self, sentences, sizes):
        # sentences - either the list of matrices or the matrix
        # sizes - either the array or the list of array
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
        # returns the next batch of the data
        # batch_size - the maximum size of the batch
        matrix = self.sentences[self.last_matrix_idx]
        # if in the matrix there is not enough data for the batch, we need to use another matrix
        if self.next_batch_idx >= len(matrix):
            self.last_matrix_idx += 1
            # if we used all the data, we need to start new epoch
            if self.last_matrix_idx >= len(self.sentences):
                self.last_matrix_idx = 0
                self.epoch_counter += 1
            self.next_batch_idx = 0
            matrix = self.sentences[self.last_matrix_idx]
        sizes = self.sizes[self.last_matrix_idx]
        from_idx = self.next_batch_idx
        to_idx = from_idx + batch_size
        batch_sentences = matrix[from_idx: to_idx]
        batch_sizes = matrix[from_idx: to_idx]
        self.next_batch_idx = to_idx
        return batch_sentences, batch_sizes

    def join_matrices(self, eos, max_size=None, shuffle=True):
        # Join all sentence matrices
        # eos - value to fill smaller matrices
        # max_size - number of columns in joint matrix
        # shuffle - whether to shuffle sentences or not
        if max_size is None:
            max_size = max(matrix.shape[1] for matrix in self.sentences)
        padded_matrices = []
        for matrix in self.sentences:
            if matrix.shape[1] == max_size:
                padded = matrix
            else:
                diff = max_size - matrix.shape[1]
                padded = np.pad(matrix, [(0, 0),(0, diff)], 'constant', constant_values=eos)
            padded_matrices.append(padded)
        sentences = np.vstack(padded_matrices)
        sizes = np.hstack(self.sizes)
        if shuffle:
            np.random.shuffle(sentences)
            np.random.shuffle(sizes)
        return sentences, sizes


def load_npz_data(path):
    # return the dataset from given .npz file
    data = np.load(path)
    if 'sentences' in data:
        return Dataset(data['sentences'], data['sizes'])
    sentences_names = sorted(name for name in data.files if name.startswith('sentences'))
    sizes_names = sorted(name for name in data.files if name.startswith('sizes'))

    sentences = []
    sizes = []

    for sentence_name, size_name in zip(sentences_names, sizes_names):
        sentences.append(data[sentence_name])
        sizes.append(data[size_name])
    return Dataset(sentences, sizes)


def load_text_data(path, word_dict):
    # read the data file and return the sentence matrix and size array
    # path - path to the file
    # word_dict - mapping from the word to the index
    max_len = 0
    all_indices = []
    sizes = []
    with open(path, 'rb') as f:
        for line in f:
            tokens = line.decode('utf-8').split()
            length = len(tokens)
            if length > max_len:
                max_len = length
            sizes.append(length)
            indices = [word_dict[token] for token in tokens]
            all_indices.append(indices)

    shape = (len(all_indices), max_len)
    sizes = np.array(sizes)
    matrix = np.full(shape, 0, np.int32)
    for i, idx in enumerate(all_indices):
        matrix[i, :len(idx)] = idx
    return matrix, sizes

