from preprocess_data import Dataset, normalizeString
from model import Model
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse


# given the beginning, generate the text
def predict(dataset, model, text, next_words=100):
    # normalize the given beginning
    words = normalizeString(text).split()
    model.eval()
    state = model.init_state(len(words))

    for i in range(0, next_words):
        # convert the text into a tensor
        x = torch.tensor([[dataset.word2idx[w] for w in words[i:]]])
        # predict the next word
        y_pred, state = model(x, state)

        last_word_logits = y_pred[0][-1]
        # get the probabilities of each word to be next
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy().astype('float64')
        # choose 3 most probable words
        idx = p.argsort()[-3:][::-1]
        # get probabilities of these words
        new_p = np.zeros(p.shape).astype('float64')
        # normalize these probabilities
        new_p[idx] = (p[idx]/np.sum(p[idx]))
        # choose the next word using these probabilities
        word_index = np.random.choice(len(last_word_logits), p=new_p)
        # if we get the end of the text then we need to stop generating
        if dataset.idx2word[word_index] == "EOS":
            break
        # add generated word to the list
        words.append(dataset.idx2word[word_index])
    # return list of generated words
    return words


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, default="state_dict_model.pt")
    parser.add_argument('--dataset', type=str, default="output/vocabulary.pkl")
    parser.add_argument('--input', type=str, default="the")

    args = parser.parse_args()
    # load the dataset
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    PATH = args.model
    # load the model
    model = Model(dataset)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    # generate the text for the given beginning
    print(' '.join(predict(dataset, model, text=args.input)))
