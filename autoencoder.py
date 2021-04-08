from torch import nn
import torch


class TextModel(nn.Module):
    # basic autoencoder class with embedding and projection layers
    def __init__(self, voc, args, init_range=0.1):
        super.__init__()
        self.voc = voc
        self.args = args
        self.embed = nn.Embedding(voc.size, args.emb_dim)
        self.proj = nn.Linear(args.dim_h, voc.size)

        # weights initialization
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform(-init_range, init_range)
