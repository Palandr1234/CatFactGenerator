import unicodedata
import re
import argparse
import os
import pickle
import pandas as pd
import torch.utils.data
from collections import Counter


# class for the Dataset
class Dataset(torch.utils.data.Dataset):
    # intialization of the dataset
    # args - arguments for the Dataset (like input folder, sequence length)
    def __init__(self, args):
        self.args = args
        # get all words in the dataset
        self.words = self.load_words()
        # get the list of all unique words in the dataset
        self.unique_words = self.get_unique_words()
        # create the dictionary for converting the index to word
        self.idx2word = {idx: word for idx, word in enumerate(self.unique_words)}
        # create the dictionary for converting the  word to index
        self.word2idx = {word: idx for idx, word in enumerate(self.unique_words)}
        # convert the dataset into the indexes
        self.word_indexes = [self.word2idx[w] for w in self.words]

    # loads the dataset and returns the list of all words
    def load_words(self):
        # read all the data and normalize it
        train_df = pd.read_csv(self.args.input, sep='\n').applymap(lambda x: normalizeString(x) + " EOS")
        # concatenate all facts in one string
        text = train_df['Fact'].str.cat(sep=' ')
        # return the array of all words in the dataset
        return text.split(' ')

    # get all unique words in the dataset
    def get_unique_words(self):
        # count appearance of all the words in the dataset
        word_cnts = Counter(self.words)
        # sort the words by the number of their appearances
        return sorted(word_cnts, key=word_cnts.get, reverse=True)

    # get the length of the dataset
    def __len__(self):
        return len(self.word_indexes) - self.args.sequence_length

    # get one item of the dataset
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


# normalize the string
def normalizeString(s):
    # convert the string to lowercase and remove spaces at the beginning and at the end
    s = unicodeToAscii(s.lower().strip())
    # put the whitespace before any appearance of symbols such as punctuation marks, opening parentheses etc
    s = re.sub(r"([),.!?%:;])", r" \1", s)
    # put the whitespace after any appearance of closing parentheses etc
    s = re.sub(r"([(])", r"\1 ", s)
    # anything that contains characters other than letters, numbers, punctuation marks, parentheses etc will be
    # substituted with the whitespace
    s = re.sub(r"[^a-zA-Z0-9%,.!?()':;-]+", r" ", s)
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Text file previously tokenized and preprocessed')
    parser.add_argument('output', help='Directory to save the data')
    parser.add_argument('--sequence-length',
                        type=int, default=6)

    args = parser.parse_args()
    dataset = Dataset(args)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    path = os.path.join(args.output, 'vocabulary.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
