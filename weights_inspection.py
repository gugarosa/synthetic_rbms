import argparse

import learnergy.visual.image as im
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Inspects both original and sampled weights and constructs a mosaic.')

    parser.add_argument(
        'original_weight', help='Input name for the original RBM weights', type=str)

    parser.add_argument(
        'sampled_weight', help='Input name for the sampled weights', type=str)

    parser.add_argument(
        'sampled_epoch', help='Epoch number for the sampled weights', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    original_weight = args.original_weight
    sampled_weight = args.sampled_weight
    sampled_epoch = args.sampled_epoch

    # Loading original weights
    weights = np.load(f'weights/{original_weight}.npy')

    # Loading sampled weights
    sampled_weights = np.load(f'weights/{sampled_weight}.npy')

    # Reshaping sampled weights to original weights size
    sampled_weights = np.reshape(sampled_weights, [sampled_weights.shape[0], weights.shape[0], weights.shape[1]])

    # Converting both arrays to tensor
    weights = torch.from_numpy(weights)
    sampled_weights = torch.from_numpy(sampled_weights[sampled_epoch-1])

    # Creating mosaics
    im.create_mosaic(weights)
    im.create_mosaic(sampled_weights)
