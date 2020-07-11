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


    def forward(self, x, hidden):


        return out, hidden


    def init_hidden(self, batch_size):


        return None

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