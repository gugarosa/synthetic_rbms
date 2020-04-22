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
    parser = argparse.ArgumentParser(usage='Reconstructs an RBM with linear combination of original and sampled weights.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'mnist', 'fmnist', 'kmnist'])

    # Adds an identifier argument to the desired pre-trained model path
    parser.add_argument(
        'input_model', help='Input name for the pre-trained RBM', type=str)

    # Adds an identifier argument to the desired weights file
    parser.add_argument(
        'input_weight', help='Input name for the weight file', type=str)

    # Adds an identifier argument to the desired sampled weights file
    parser.add_argument(
        'input_sampled', help='Input name for the sampled weight file', type=str)

    # Adds an identifier argument to the desired  file
    parser.add_argument(
        '-alpha', help='Constant used to calculate the linear combination between weights', type=float, default=0.01)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_model = args.input_model
    input_weight = args.input_weight
    input_sampled = args.input_sampled
    alpha = args.alpha

    # Loads the testing data
    _, test = s.load_dataset(name=dataset)

    # Loads the pre-trained model
    model = torch.load(f'models/{input_model}.pth')

    # Loading original and sampled weights
    W = np.load(f'weights/{input_weight}.npy')
    W_sampled = np.load(f'weights/{input_sampled}.npy')

    # Reshaping weights to correct dimension
    W = np.reshape(W, [model.n_visible, model.n_hidden])
    W_sampled = np.reshape(W_sampled, [model.n_visible, model.n_hidden])

    # Resetting biases for fair comparison
    model.a = torch.nn.Parameter(torch.zeros(model.n_visible))
    model.b = torch.nn.Parameter(torch.zeros(model.n_hidden))

    # Applying linear combination of original and sampled weights as new weights
    model.W = torch.nn.Parameter((1 - alpha) * torch.from_numpy(W) + alpha * torch.from_numpy(W_sampled))

    # Checking model device type
    if model.device == 'cuda':
        # Applying its parameters as cuda again
        model = model.cuda()

    # Reconstructs an RBM
    mse, pl = model.reconstruct(test)
