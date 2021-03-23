from collections import Counter, defaultdict
import math
import numpy as np


def create_sentence_matrix(path: str, num_sentences: int, min_threshold: int, max_threshold: int, word_dict: dict) \
        -> [np.array, np.array]:
    # creates a sentence matrix for the file
    # path - path for the file
    # num_sentences - number of sentences in the given range of lengths
    # min_threshold - the lower bound of the range of lengths
    # max_threshold - the upper bound of the range of lengths
    # word_dict - mapping from the word to its number
    sentence_matrix = np.zeros((num_sentences, max_threshold))
    sizes = np.empty(num_sentences)
    i = 0
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            tokens = line.split()
            size = len(tokens)
            # if the size of the sentence is not in the range we need then discard it
            if size < min_threshold or size > max_threshold:
                continue
            # map every word from a sentence into a number
            array = np.array([word_dict[token] for token in tokens])
            sentence_matrix[i, :size] = array
            sizes[i] = size
            i += 1
    return sentence_matrix, sizes


def load_data(path: str, max_size: int, min_num: int = 5, test_proportion: float = 0.1) -> [dict, dict, list]:
    # creates a train sentence matrix, a test sentence matrix and a vocabulary for the file
    # path - path for the file
    # max_size - maximum length of the sentence
    # min_num - minimum number the word needs to appear in the text to be in the vocabulary
    # test_proportion - proportion of the test set
    token_counter = Counter()
    size_counter = Counter()
    # creating a vocabulary and counting the sentence sizes
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            tokens = line.split()
            size = len(tokens)
            if size > max_size:
                continue
            # keep track of different size ranges
            level = int(math.ceil(size/10) * 10)
            size_counter[level] += 1
            token_counter.update(tokens)
    # sort the vocabulary and discard all the words which has the number of occurrences less than min_num
    voc = [w for w, count in token_counter.most_common() if count >= min_num]
    mapping = zip(voc, range(len(voc)))
    print(voc)

    d = defaultdict(lambda: 1, mapping)

    # create the sentence matrix
    train_data = {}
    test_data = {}
    for threshold in size_counter:
        min_threshold = threshold - 9
        num_sentences = size_counter[threshold]
        sentences, sizes = create_sentence_matrix(path, num_sentences, min_threshold, threshold, d)
        # shuffle the sentence matrix
        np.random.shuffle(sentences)
        np.random.shuffle(sizes)
        ind = int(len(sentences) * test_proportion)
        test_sentences = sentences[:ind]
        test_sizes = sizes[:ind]
        train_sentences = sentences[ind:]
        train_sizes = sizes[ind:]
        train_data[f'sentences-{threshold}'] = train_sentences
        train_data[f'sizes-{threshold}'] = train_sizes
        test_data[f'sentences-{threshold}'] = test_sentences
        test_data[f'sizes-{threshold}'] = test_sizes
    return train_data, test_data, voc


if __name__ == '__main__':
    load_data("putin.txt", 100)
