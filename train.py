import pickle
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from preprocess_data import Dataset
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(args.max_epoch)):
        state = model.init_state(args.sequence_length)
        losses = []
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            y_pred, state = model(x, state)
            loss = criterion(y_pred.transpose(1, 2), y)
            state = state.detach()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(loss.item())
        print({'epoch': epoch, 'loss': sum(losses)/len(losses)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sequence_length', type=int, default=6)
    parser.add_argument('--dataset_path', type=str, default='output/vocabulary.pkl')
    parser.add_argument('--save_path', type=str, default='state_dict_model.pt')
    args = parser.parse_args()

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    model = Model(dataset).to(device)
    train(dataset, model, args)
    PATH = args.save_path

    torch.save(model.state_dict(), PATH)
