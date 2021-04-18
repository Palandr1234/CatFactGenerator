import pickle
import torch
import argparse
from tqdm import tqdm
import random
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from preprocess_data import Dataset
from autoencoder import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in tqdm(range(args.max_epoch)):
        state_h, state_c = model.init_state(args.sequence_length)
        losses = []
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)
            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print({'epoch': epoch, 'loss': sum(losses)/len(losses)})


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sequence_length', type=int, default=4)
args = parser.parse_args()

with open('output/vocabulary.pkl', 'rb') as f:
    dataset = pickle.load(f)

model = Model(dataset).to(device)
train(dataset, model, args)
