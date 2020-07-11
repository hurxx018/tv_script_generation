

def create_lookup_table(
    text
    ):
    """
    """
    unique_words = set(text)
    int_to_vocab = dict(enumerate(unique_words))
    vocab_to_int = {word:i for i, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab
