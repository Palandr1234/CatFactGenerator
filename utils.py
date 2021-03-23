from collections import defaultdict


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


create_word_list('output/vocabulary.txt')

