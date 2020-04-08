import torchvision

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': torchvision.datasets.MNIST,
    'fmnist': torchvision.datasets.FashionMNIST,
    'kmnist': torchvision.datasets.KMNIST
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
                           transform=torchvision.transforms.ToTensor())

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=torchvision.transforms.ToTensor())

    return train, test
