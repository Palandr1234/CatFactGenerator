from torch import nn


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.hidden_size = 128
        self.embed_size = 128
        self.nlayers = 3
        n_vocab = len(dataset.unique_words)
        self.embed = nn.Embedding(n_vocab, self.embed_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.nlayers, dropout=0.2)
        self.fc = nn.Linear(self.hidden_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embed(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state
