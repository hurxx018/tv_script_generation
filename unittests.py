
def _print_success_message():
    print('Tests Passed')

def test_create_lookup_table(
    create_lookup_table
    ):
    test_text = '''
        Moe_Szyslak Moe's Tavern Where the elite meet to drink
        Bart_Simpson Eh yeah hello is Mike there Last name Rotch
        Moe_Szyslak Hold on I'll check Mike Rotch Mike Rotch Hey has anybody seen Mike Rotch lately
        Moe_Szyslak Listen you little puke One of these days I'm gonna catch you and I'm gonna carve my name on your back with an ice pick
        Moe_Szyslak Whats the matter Homer You're not your normal effervescent self
        Homer_Simpson I got my problems Moe Give me another one
        Moe_Szyslak Homer hey you should not drink to forget your problems
        Barney_Gumble Yeah you should only drink to enhance your social skills'''

    test_text = test_text.lower().split()

    vocab_to_int, int_to_vocab = create_lookup_table(test_text)

    # Check types
    assert isinstance(vocab_to_int, dict), "vocab_to_int is not a dictionary"
    assert isinstance(int_to_vocab, dict), "int_to_vocab is not a dictionary"

    # Compare lengths
    assert len(vocab_to_int) == len(int_to_vocab), \
        """Lengths of vocab_to_int and int_to_vocab do not match. 
        vocab_to_int has a length of {:d} and int_to_vocab has a length of {:d}""".format(len(vocab_to_int), len(int_to_vocab))

    # Test if the two dictionaries have the same words and indices.
    vocab_to_int_word_set = set(vocab_to_int.keys())
    int_to_vocab_word_set = set(int_to_vocab.values())
    assert not (vocab_to_int_word_set - int_to_vocab_word_set), \
        """vocab_to_int and int_to_vocab do not have the same words.
        {} found in vocab_to_int, but not in int_to_vocab.
        """.format(vocab_to_int_word_set - int_to_vocab_word_set)
    assert not (int_to_vocab_word_set - vocab_to_int_word_set), \
        """vocab_to_int and int_to_vocab do not have the same words.
        {} found in int_to_vocab, but not in vocab_to_int.
        """.format(int_to_vocab_word_set - vocab_to_int_word_set)

    vocab_to_int_indices_set = set(vocab_to_int.values())
    int_to_vocab_indices_set = set(int_to_vocab.keys())
    assert not (vocab_to_int_indices_set - int_to_vocab_indices_set), \
        """vocab_to_int and int_to_vocab do not have the same indices.
        {} found in vocab_to_int, but not in int_to_vocab.
        """.format(vocab_to_int_indices_set - int_to_vocab_indices_set)
    assert not (int_to_vocab_indices_set - vocab_to_int_indices_set), \
        """vocab_to_int and int_to_vocab do not have the same indices.
        {} found in int_to_vocab, but not in vocab_to_int.
        """.format(int_to_vocab_indices_set - vocab_to_int_indices_set)

    # Test if the two dict make the same lookup.
    miss_matches = [(word, id, id, int_to_vocab[id]) for word, id in vocab_to_int.items() if int_to_vocab[id] != word]

    assert not miss_matches, \
        "Found {} missmatche(s). First missmatch: vocab_to_int[{}] = {} and int_to_vocab[{}] = {}".format(
            len(miss_matches),
            *miss_matches[0])

    assert len(vocab_to_int) > len(set(test_text))/2,\
        "The length of vocab seems too small.  Found a length of {}".format(len(vocab_to_int))

    _print_success_message()