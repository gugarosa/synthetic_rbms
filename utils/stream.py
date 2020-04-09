import numpy as np
import tensorflow as tf
import torchvision as tv

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': tv.datasets.MNIST,
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST
}


def load_dataset(name='mnist'):
    """Loads an input dataset.

    Args:
        name (str): Name of dataset to be loaded.

    Returns:
        Training and testing sets of loaded dataset.

    """

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.ToTensor())

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.ToTensor())

    return train, test

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
