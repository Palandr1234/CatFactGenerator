import unicodedata
import re
import argparse
import os
import pickle


class WordDictionary:
    def __init__(self):
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2cnt[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2cnt[word] += 1


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


def load_data(path: str, max_size: int) -> [dict, dict, list]:
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    voc = WordDictionary()
    sentences = [normalizeString(l) for l in lines]
    for sentence in sentences:
        if len(sentence.split()) < max_size:
            voc.add_sentence(sentence)
    return voc, sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Text file previously tokenized and preprocessed')
    parser.add_argument('output', help='Directory to save the data')
    parser.add_argument('--max-length',
                        help='Maximum sentence length (default 100)',
                        type=int, default=100, dest='max_length')

    args = parser.parse_args()

    voc, sentences = load_data(args.input, args.max_length)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    path = os.path.join(args.output, 'vocabulary.pkl')
    with open(path, 'wb') as f:
        pickle.dump(voc, f)

    path = os.path.join(args.output, 'sentences.txt')
    text = '\n'.join(sentences)
    with open(path, 'wb') as f:
        f.write(text.encode('utf-8'))
