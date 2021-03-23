from torch import nn
import torch


class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super.__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=64)
        self.decoder_hidden_layer = nn.Linear(in_features=64, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=kwargs["input_shape"])

    def forward(self, features):
        activation = torch.relu(self.encoder_hidden_layer(features))
        activation = torch.relu(self.encoder_output_layer(activation))
        activation = torch.relu(self.decoder_hidden_layer(activation))
        activation = torch.relu(self.decoder_output_layer(activation))
        return activation
