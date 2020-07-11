import os



def load_data(filename):

    with open(filename, 'r') as f:
        text = f.read()
    return text