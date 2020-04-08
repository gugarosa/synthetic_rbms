import numpy as np


def save_as_numpy(t, output_file=''):
    """Saves a PyTorch tensor into a Numpy array.

    Args:

    """

    #
    t_numpy = t.detach().cpu().numpy()

    #
    np.save(output_file, t_numpy)
