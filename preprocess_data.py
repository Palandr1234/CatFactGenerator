import unicodedata
import re
import argparse
import os
import pickle
import pandas as pd
import torch.utils.data
from collections import Counter


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.words = self.load_words()
        self.unique_words = self.get_unique_words()
        self.idx2word = {idx: word for idx, word in enumerate(self.unique_words)}
        self.word2idx = {word: idx for idx, word in enumerate(self.unique_words)}

        self.word_indexes = [self.word2idx[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv(self.args.input, sep='\n').applymap(lambda x: normalizeString(x))
        text = train_df['Fact'].str.cat(sep=' ')
        return text.split(' ')

    def get_unique_words(self):
        word_cnts = Counter(self.words)
        return sorted(word_cnts, key=word_cnts.get, reverse=True)

    def __len__(self):
        return len(self.word_indexes) - self.args.sequence_length

    def __getitem__(self, item):
        return (
            torch.tensor(self.word_indexes[item:item+self.args.sequence_length]),
            torch.tensor(self.word_indexes[item+1: item+self.args.sequence_length+1])
        )


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Text file previously tokenized and preprocessed')
    parser.add_argument('output', help='Directory to save the data')
    parser.add_argument('--sequence-length',
                        type=int, default=4)

    args = parser.parse_args()
    dataset = Dataset(args)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    path = os.path.join(args.output, 'vocabulary.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
