import argparse

import numpy as np

import torch
import utils.stream as s


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
        'input_model', help='Input name for the pre-trained RBM', type=str)

    # Adds an identifier argument to the desired weights file
    parser.add_argument(
        'input_weight', help='Input name for the weight file', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_model = args.input_model
    input_weight = args.input_weight

    # Loads the testing data
    _, test = s.load_dataset(name=dataset)

    # Loads the pre-trained model
    model = torch.load(f'models/{input_model}.pth')

    # Loading sampled weights
    W = np.load(f'weights/{input_weight}.npy')

    # Reshaping weights to correct dimension
    W = np.reshape(W, [model.n_visible, model.n_hidden])

    # Resetting biases for fair comparison
    model.a = torch.nn.Parameter(torch.zeros(model.n_visible))
    model.b = torch.nn.Parameter(torch.zeros(model.n_hidden))

    # Applying loaded weights as new weights
    model.W = torch.nn.Parameter(torch.from_numpy(W))

    # Checking model device type
    if model.device == 'cuda':
        # Applying its parameters as cuda again
        model = model.cuda()

    # Fits an RBM
    mse, pl = model.reconstruct(test)
