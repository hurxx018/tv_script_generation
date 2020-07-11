import os
import pickle


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

