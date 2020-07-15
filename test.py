import unittests
import torch
from preprocess import create_lookup_table, token_lookup
from neural_networks import batch_data, RNN, forward_back_prop

unittests.test_create_lookup_table(create_lookup_table)

unittests.test_token_lookup(token_lookup)

unittests.test_batch_data(batch_data)

train_on_gpu = torch.cuda.is_available()
unittests.test_rnn(RNN, train_on_gpu)

unittests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)