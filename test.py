from preprocess_data import Dataset, normalizeString
from model import Model
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse


def predict(dataset, model, text, next_words=100):
    words = normalizeString(text).split()
    model.eval()
    state = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word2idx[w] for w in words[i:]]])
        y_pred, state = model(x, state)

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy().astype('float64')
        idx = p.argsort()[-5:][::-1]
        new_p = np.zeros(p.shape).astype('float64')
        new_p[idx] = (p[idx]/np.sum(p[idx]))
        # new_p[idx] /= np.sum(new_p)
        word_index = np.random.choice(len(last_word_logits), p=new_p)
        if dataset.idx2word[word_index] == "EOS":
            break
        words.append(dataset.idx2word[word_index])

    return words


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, default="state_dict_model.pt")
    parser.add_argument('--dataset', type=str, default="output/vocabulary.pkl")
    parser.add_argument('--input', type=str, default="the")

    args = parser.parse_args()
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    PATH = args.model
    model = Model(dataset)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    print(' '.join(predict(dataset, model, text=args.input)))



