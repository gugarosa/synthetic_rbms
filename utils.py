import numpy as np


def save_as_numpy(t, output_file=''):
    """Saves a PyTorch tensor into a Numpy array.

    Args:
        t (torch.Tensor): A PyTorch tensor.
        output_file (str): Name for the output file.

    """

    # Transforms the tensor to a numpy array
    t_numpy = t.detach().cpu().numpy()

    # Saves the numpy array
    np.save(output_file, t_numpy)
