import pickle
import torch
import argparse
from preprocess_data import WordDictionary
from autoencoder import VAE

SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indexesFromSentence(voc, sentence):
    return [voc.word2idx[word] for word in sentence.split()]


def tensorFromSentence(voc, sentence):
    idx = indexesFromSentence(voc, sentence)
    idx.append(EOS_token)
    return torch.tensor(idx, dtype=torch.long, device=device)


def getSentence(tensor, voc):
    decoded_sentence = []
    for i in range(tensor.shape[0]):
        topvalue, topindex = tensor[i, :].data.topk(1)
        if topindex.item() == EOS_token:
            decoded_sentence.append('<EOS>')
        else:
            decoded_sentence.append(voc.idx2word[topindex.item()])
    return decoded_sentence


def train(input_tensor, target_tensor, vae, optimizer, criterion, max_length):

    optimizer.zero_grad()

    input_tensor.to(device)
    target_tensor.to(device)

    output_tensor = vae(input_tensor)
    loss = criterion(target_tensor, output_tensor)
    loss.backward()
    optimizer.step()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('embed_size', type=int)
parser.add_argument('hidden_size', type=int)
parser.add_argument('nlayers', type=int)
parser.add_argument('dropout', type=float)
parser.add_argument('latent_dim', type=int)
parser.add_argument('--max_length', type=int, default=100)
args = parser.parse_args()

lines = open('output/sentences.txt', encoding='utf-8').read().strip().split('\n')
sentences = [line for line in lines]
with open('output/vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)
print(voc.n_words)
vae = VAE(voc.n_words, args)
for sentence in sentences:
    tensor = tensorFromSentence(voc, sentence)
    print(tensor)
    output = vae(tensor)
    print(output.shape)
    print(getSentence(output, voc))
    break

