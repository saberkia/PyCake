import numpy as np
import idx2numpy


def load_npz_data(address):
    """Load NZP data and build train and test sets.

    Parameters
    ----------
    address : str
        address of .npz file

    Returns
    -------
    numpy array
        train and test set, both data and label .
    """
    data = np.load(address)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)


def load_ubyte_data(address):
    """Load NZP data and build train and test sets.

    Parameters
    ----------
    address : str
        address of .ubyte file

    Returns
    -------
    list
    """
    array = idx2numpy.convert_from_file(address)
    return array


def load_ubyte():
    return None


def normalize_data(train, test):
    """Normalizing both train and test sets Simultaneously .

    Parameters
    ----------
    train : numpy array

    test : numpy array

    Returns
    -------
    numpy array
        normalized train and test set .
    """
    return train / 255.0, test / 255.0


def reshape_data(train, test):
    """Normalizing both train and test sets Simultaneously .

    Parameters
    ----------
    train : numpy array

    test : numpy array

    Returns
    -------
    numpy array
        reshaped train and test set .
    """
    x_train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
    x_test = test.reshape(test.shape[0], test.shape[1], test.shape[2], 1)
    return x_train, x_test
