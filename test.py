import unittests

from preprocess import create_lookup_table, token_lookup
from neural_networks import batch_data

unittests.test_create_lookup_table(create_lookup_table)

unittests.test_token_lookup(token_lookup)

unittests.test_batch_data(batch_data)