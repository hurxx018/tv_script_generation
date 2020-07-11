import os



def load_data(filename):

    path_name = os.path.join(filename)
    with open(path_name, 'r') as f:
        text = f.read()
    return text