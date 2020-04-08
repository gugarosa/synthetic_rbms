import argparse

import numpy as np
import torch

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Reconstructs an RBM with loaded weights.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'mnist', 'fmnist', 'kmnist'])

    # Adds an identifier argument to the desired pre-trained model path
    parser.add_argument(
        'input', help='Input file for the pre-trained RBM', type=str)

    # Adds an identifier argument to the desired weights file
    parser.add_argument(
        'weights', help='Input file for the weights', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_file = args.input
    input_weights = args.weights

    # Loads the testing data
    _, test = l.load_dataset(name=dataset)

    # Loads the pre-trained model
    model = torch.load(input_file)

    # Loading extracted weights
    W = np.load(input_weights)

    # Applying loaded weights as new weights
    model.W = torch.nn.Parameter(torch.from_numpy(W))

    # Fits an RBM
    mse, pl = model.reconstruct(test)
