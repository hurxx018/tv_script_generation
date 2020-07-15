import os

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
        """ Initialize RNN
            Arguments:
            vocab_size : Number of words in the vocabulary
            output_size : Number of words in the vocabulary
            embedding_dim : Dimension of Embedding Layer
            hidden_dim : Dimension of hidden state of LSTM
            n_layers : Number of Layers of LSTM
            dropout : Probability of Dropout
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
        """ Forward
            Arguments:
            nn_input : input data with shape (Batch_size, sequence_length)
            hidden : hidden state tuple of two elements

            Returns:
            outputs :
            hidden  : hidden state
        """
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

def weights_init_normal(
    m
    ):
    """ Initialize weights of Linear layer with normal distribution
    """
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1./np.sqrt(n)
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
        n, _ = m.weight.shape
        y = 1./np.sqrt(n)
        m.weight.data.normal_(0, y)

def batch_data(
    words, 
    sequence_length,
    batch_size, 
    validation_fraction = 0.1
    ):
    """ Generate batch_data: train_loader and valid_loader
        Arguments
        ---------
        words: a sequence of words
        sequence_length: Length of sequence
        batch_size : Number of sequence per batch
        validation_fraction : Portion of validation data

        Returns
        -------
        train_loader
        valid_loader
    """
    words = np.asarray(words, dtype=np.int64)

    total_number_of_sequences = words.size - sequence_length

    # Inputs and Targets
    x = np.empty((total_number_of_sequences, sequence_length), dtype = words.dtype)
    y = np.empty(total_number_of_sequences, dtype = words.dtype)
    for i in range(words.size - sequence_length):
        x[i, :] = words[i : i + sequence_length]
        y[i]    = words[i + sequence_length]

    n_train = np.round((1. - validation_fraction)*len(x)).astype(np.int64)
    rng = np.random.default_rng()
    choice = rng.choice(range(len(x)), size=(n_train,), replace=False)
    ind = np.zeros(len(x), dtype=bool)
    ind[choice] = True
    rest = ~ind

    train_x, train_y, valid_x, valid_y = x[ind], y[ind], x[rest], y[rest]

    train_data = TensorDataset(torch.LongTensor(train_x), torch.LongTensor(train_y))
    train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size)

    valid_data = TensorDataset(torch.LongTensor(valid_x), torch.LongTensor(valid_y))
    valid_loader = DataLoader(valid_data, shuffle = True, batch_size = batch_size)

    return train_loader, valid_loader



def forward_back_prop(
    rnn,
    optimizer,
    criterion,
    inputs,
    targets,
    hidden,
    clip,
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

    # 'clip_grad_norm' helps prevent the exploding gradient problem
    # in RNNs
    nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()

    # return the loss over a bach and the hidden state
    # produced by our model
    return loss.item(), hidden


def train(
    rnn,
    train_loader,
    valid_loader,
    batch_size,
    optimizer,
    criterion,
    n_epochs,
    clip = 5.,
    show_every_n_batches = 100,
    train_on_gpu = False,
    savecheckpoint = None
    ):
    """ Train
    """
    batch_losses = []

    if train_on_gpu:
        rnn.cuda()

    rnn.train()
    min_validation_loss = np.Infinity
    print("Training for {:d} epoch(s)...".format(n_epochs))
    for epoch_i in range(1, n_epochs + 1):
        # initialize hidden
        hidden = rnn.init_hidden(batch_size, train_on_gpu)

        for batch_i, (inputs, targets) in enumerate(train_loader, 1):
            # Make sure that the size of inputs equals batch_size.
            # otherwise, the size of hidden state does not match to the size of inputs.
            if len(inputs) != batch_size:
                break
            # forward and backpropagation
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, targets, hidden, clip, train_on_gpu)
            # record loss
            batch_losses.append(loss)

            # print training loss stats and evaluate the validation set.
            if (batch_i % show_every_n_batches == 0):

                # Evaluate the Validation dataset
                rnn.eval()
                valid_losses = []
                valid_hidden = rnn.init_hidden(batch_size, train_on_gpu)
                for valid_batch_j, (inputs, targets) in enumerate(valid_loader, 1):
                    # Make sure that the size of valid_inputs equals batch_size.
                    if len(inputs) != batch_size:
                        break
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    valid_hidden = tuple([h.data for h in valid_hidden])
                    outputs, valid_hidden = rnn(inputs, valid_hidden)
                    loss = criterion(outputs, targets)

                    valid_losses.append(loss.item())

                mean_valid_losses = np.mean(valid_losses)
                print("Epoch: {:>4}/{:<4}....Step: {}....Train_Loss: {}....Valid_Loss: {}....Min Valid Loss: {}\n".format(
                    epoch_i, n_epochs, batch_i, np.mean(batch_losses), mean_valid_losses, min_validation_loss
                ))

                if min_validation_loss > mean_valid_losses:
                    min_validation_loss = mean_valid_losses
                    if isinstance(savecheckpoint, str):
                        print("Saving the RNN model.\n")
                        save_rnn_model(savecheckpoint, rnn)


                batch_losses = []                
                rnn.train()

    # returns a trained rnn
    return rnn



def save_rnn_model(
    filename, 
    decoder
    ):
    """ Save RNN model
        Arguments
        ---------
        filename : checkpoint filename
        decoder : RNN model

        Returns:
        None
    """
    checkpoint = {
        "vocab_size"    : decoder.vocab_size,
        "output_size"   : decoder.output_size,
        "hidden_dim"    : decoder.hidden_dim,
        "embedding_dim" : decoder.embedding_dim,
        "n_layers"      : decoder.n_layers,
        "drop_prob"     : decoder.drop_prob,
        "state_dict"    : decoder.state_dict()
        }
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(checkpoint, save_filename)


def load_rnn_model(
    filename
    ):
    """ Load RNN model
        Arguments
        ---------
        filename : checkpoint filename

        Returns
        -------
        decoder : RNN 
    """
    load_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'

    with open(load_filename, 'rb') as f:
        checkpoint = torch.load(load_filename)

    decoder = RNN(
        checkpoint["vocab_size"],
        checkpoint["output_size"],
        checkpoint["hidden_dim"],
        checkpoint["embedding_dim"],
        checkpoint["n_layers"],
        checkpoint["drop_prob"]
        )

    decoder.load_state_dict(checkpoint["state_dict"])
    return decoder