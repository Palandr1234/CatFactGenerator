from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, input_size, args):
        super(VAE, self).__init__()
        self.embed = nn.Embedding(input_size, args.embed_size)
        self.lstm1 = nn.LSTM(args.embed_size, args.hidden_size, args.nlayers, dropout=args.dropout, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)
        self.h2mu = nn.Linear(args.hidden_size * args.nlayers * 2, args.latent_dim)
        self.h2logvar = nn.Linear(args.hidden_size * args.nlayers * 2, args.latent_dim)
        self.z2emb = nn.Linear(args.latent_dim, args.embed_size)
        self.lstm2 = nn.LSTM(args.embed_size, args.hidden_size, args.nlayers, dropout=args.dropout)
        self.proj = nn.Linear(args.hidden_size, input_size)
        self.max_length = args.max_length

    def encode(self, input):
        input = self.embed(input).view(len(input), 1, -1)
        _, (h, _) = self.lstm1(input)
        h = self.dropout(h.view(1, -1))
        return self.h2mu(h), self.h2logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, hidden=None):
        z = z.repeat(self.max_length, 1)
        z = self.z2emb(z)
        z = z.view(len(z), 1, -1)
        output, _ = self.lstm2(z, hidden)
        output = self.dropout(output).view(-1, output.size(-1))
        output = self.proj(output)
        output = nn.LogSoftmax(dim=1)(output)
        return output

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)





