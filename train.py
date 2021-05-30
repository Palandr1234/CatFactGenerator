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


# train the given model
# dataset - the dataset for training
# model - the model to be trained
# args - other arguments such as the number of epochs, batch size etc.
def train(dataset, model, args):
    model.train()

    # create the dataloader for the dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    # initialize the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(args.max_epoch)):
        # initialize the state of the model
        state = model.init_state(args.sequence_length)
        losses = []
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            # get the output for the given input and state
            y_pred, state = model(x, state)
            # calculate the loss
            loss = criterion(y_pred.transpose(1, 2), y)
            # get the new state detached from the computational graph
            state = state.detach()
            # calculate the gradients
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # update the parameters based on the current gradient
            losses.append(loss.item())
        print({'epoch': epoch, 'loss': sum(losses)/len(losses)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--dataset_path', type=str, default='output/vocabulary.pkl')
    parser.add_argument('--save_path', type=str, default='state_dict_model.pt')
    args = parser.parse_args()

    # load the dataset
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # train the model
    model = Model(dataset).to(device)
    train(dataset, model, args)
    PATH = args.save_path

    # save the model
    torch.save(model.state_dict(), PATH)
