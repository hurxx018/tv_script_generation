import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

class RNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        dropout = 0.5
        ):
        """
        """
        super(RNN, self).__init__()

        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = dropout

        # define layers
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first = True, dropout = dropout)
        
        self.fc1 = nn.Linear(hidden_dim, output_size)

        self.dropout = nn.Dropout(dropout)


    def forward(
        self, 
        nn_input, 
        hidden
        ):

        batch_size, seq_length = nn_input.shape
        nn_input = nn_input.long()
        x = self.embed(nn_input)
        x, hidden = self.lstm(x, hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        out = self.fc1(x)
        out = out.view(-1, seq_length, self.output_size)
        # return one batch of output word scores and the hidden state
        out = out[:, -1]

        return out, hidden


    def init_hidden(
        self, 
        batch_size, 
        train_on_gpu = False
        ):
        """ Initialize hidden state of LSTM
            two hidden states: hidden state and cell state of LSTM
        """
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
            )

        return hidden



def batch_data(
    words, 
    sequence_length,
    batch_size
    ):
    """
    """
    words = np.asarray(words, dtype=int)

    total_number_of_sequences = words.size - sequence_length

    # Inputs and Targets
    x = np.empty((total_number_of_sequences, sequence_length), dtype = words.dtype)
    y = np.empty(total_number_of_sequences, dtype = words.dtype)
    for i in range(words.size - sequence_length):
        x[i, :] = words[i : i + sequence_length]
        y[i]    = words[i + sequence_length]

    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(data, shuffle = True, batch_size = batch_size)
    return loader



def forward_back_prop(
    rnn,
    optimizer,
    criterion,
    inputs,
    targets,
    hidden,
    train_on_gpu = False
    ):
    """ Define Forward and Backpropagation
    """

    # move data to GPU if available
    if train_on_gpu:
        inputs, targets = inputs.cuda(), targets.cuda()

    # Creating new variables for the hidden state, otherwise
    # we would backprop through the entire training history
    hidden = tuple([h.data for h in hidden])

    outputs, hidden = rnn(inputs, hidden)

    loss = criterion(outputs, targets)

    # perform backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # return the loss over a bach and the hidden state
    # produced by our model
    return loss.item(), hidden