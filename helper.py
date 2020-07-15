import os
import pickle
import torch

from neural_networks import RNN

def load_data(
    filename
    ):
    """ Load data from file
        Arguments:
        filename : a string of filename

        Returns:
        a text string
    """
    path_name = os.path.join(filename)
    with open(path_name, 'r') as f:
        text = f.read()
    return text


def preprocess_and_save_data(
    dataset_path,
    token_lookup,
    create_lookup_tables,
    output_path
    ):
    """ Preprocess text data and Pickle the result
        Arguments:
        dataset_path: a filename of text
        token_lookup: preprocess.token_lookup
        create_lookup_tables: preprocess.create_lookup_tables
        output_path: a picklefile name for the output

        Returns:
        None
    """
    SPECIAL_WORDS = {'PADDING': '<PAD>'}

    text = load_data(dataset_path)
    
    # Ignore notice, since we do not use it for analyzing the data
    text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, " {} ".format(token))

    text = text.lower().split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]

    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict),
    open(output_path, 'wb'))



def load_preprocess(
    preprocessed_filename
    ):
    """ Load the preprocessed Training data and
        Return them 
    """
    return pickle.load(open(preprocessed_filename, 'rb'))



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