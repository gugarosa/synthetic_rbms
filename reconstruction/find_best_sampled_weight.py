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
    parser = argparse.ArgumentParser(
        usage='Finds the best sampled weight by reconstructing an RBM over a validation set with original and sampled weights.')

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

    parser.add_argument(
        '-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_model = args.input_model
    input_weight = args.input_weight
    input_sampled = args.input_sampled
    seed = args.seed

    # Loads the validation data
    _, val, _ = s.load_dataset(name=dataset, seed=seed)

    # Loads the pre-trained model
    model = torch.load(f'models/{input_model}.pth')

    # Loading original and sampled weights
    W = np.load(f'weights/{input_weight}.npy')
    W_sampled = np.load(f'weights/{input_sampled}.npy')

    # Reshaping weights to correct dimension
    W = np.reshape(W, [model.n_visible, model.n_hidden])
    W_sampled = np.reshape(W_sampled, [W_sampled.shape[0], model.n_visible, model.n_hidden])

    # Resetting biases for fair comparison
    model.a = torch.nn.Parameter(torch.zeros(model.n_visible))
    model.b = torch.nn.Parameter(torch.zeros(model.n_hidden))

    # Applying original weights
    model.W = torch.nn.Parameter(torch.from_numpy(W))

    # Checking model device type
    if model.device == 'cuda':
        # Applying its parameters as cuda again
        model = model.cuda()

    # Reconstructs the original RBM
    original_mse, _ = model.reconstruct(val)

    # Defining best sampled MSE as a high value
    best_sampled_mse = 9999999

    # Iterating over all possible epochs
    for e in range(W_sampled.shape[0]):
        print(f'Weights from GAN epoch {e+1}/{W_sampled.shape[0]}')

        # Resetting biases for fair comparison
        model.a = torch.nn.Parameter(torch.zeros(model.n_visible))
        model.b = torch.nn.Parameter(torch.zeros(model.n_hidden))

        # Applying sampled weights
        model.W = torch.nn.Parameter(torch.from_numpy(W_sampled[e]))

        # Checking model device type
        if model.device == 'cuda':
            # Applying its parameters as cuda again
            model = model.cuda()

        # Reconstructs an RBM
        sampled_mse, _ = model.reconstruct(val)

        # Checking if current sampled MSE was better than previous one
        if sampled_mse < best_sampled_mse:
            # Saving best MSE and best epoch values
            best_sampled_mse, best_epoch = sampled_mse, e

    print(f'Validation finished and best RBM found.')
    print(f'Original MSE: {original_mse} | Best Sampled MSE: {best_sampled_mse} | Epoch: {best_epoch+1}')
