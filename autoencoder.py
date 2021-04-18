from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.hidden_size = 128
        self.embed_size = 128
        self.n_layers = 3
        n_vocab = len(dataset.unique_words)
        self.embed = nn.Embedding(n_vocab, self.embed_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, dropout=0.2)
        self.fc = nn.Linear(self.hidden_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embed(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.n_layers, sequence_length, self.hidden_size),
                torch.zeros(self.n_layers, sequence_length, self.hidden_size))

