import os



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