import numpy as np
import tensorflow as tf
import torch
import torchvision as tv

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': tv.datasets.MNIST,
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST
}


def load_dataset(name='mnist', val_split=0.2, seed=0):
    """Loads an input dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.
        seed (int): Seed for splitting the training and validation data.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.ToTensor())

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.ToTensor())

    return train, val, test


def save_torch_as_numpy(t, output_file=''):
    """Saves a PyTorch tensor into a Numpy array.

    Args:
        t (torch.Tensor): A PyTorch tensor.
        output_file (str): Name for the output file.

    """

    # Transforms the tensor to a numpy array
    t_numpy = t.detach().cpu().numpy()

    # Saves the numpy array
    np.save(output_file, t_numpy)


def save_tf_as_numpy(t, output_file=''):
    """Saves a Tensorflow tensor into a Numpy array.

    Args:
        t (tf.Tensor): A PyTorch tensor.
        output_file (str): Name for the output file.

    """

    # Checks if it is a four-dimensional tensor
    if len(t.shape) == 4:
        # If yes, squeezes second and third dimensions
        t = tf.squeeze(tf.squeeze(t, 1), 1)

    # Transforms the tensor to a numpy array
    t_numpy = t.numpy()

    # Saves the numpy array
    np.save(output_file, t_numpy)
