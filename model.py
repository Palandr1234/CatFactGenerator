from torch import nn
import torch
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.hidden_size = 512
        self.embed_size = 200
        self.n_layers = 4
        n_vocab = len(dataset.unique_words)
        self.embed = nn.Embedding(n_vocab, self.embed_size)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.hidden_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embed(x)
        output, state = self.gru(embed, prev_state)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return Variable(torch.zeros(self.n_layers, sequence_length, self.hidden_size))
