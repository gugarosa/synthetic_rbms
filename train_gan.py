import argparse

import numpy as np
import tensorflow as tf
from nalp.datasets.image import ImageDataset
from nalp.models.gan import GAN


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Trains a GAN using RBM-extracted weight vectors.')

    # Adds an identifier argument to the desired extracted weights
    parser.add_argument(
        'input', help='Input file for the extracted weights', type=str)

    # Adds an identifier argument to the desired number of training epochs
    parser.add_argument(
        'epochs', help='Number of training epochs', type=int)

    # Adds an identifier argument to the desired output model file
    parser.add_argument(
        'output', help='Output file for saved model', type=str)

    # Adds an identifier argument to the desired noise dimension
    parser.add_argument(
        '-noise', help='Noise dimension', type=int, default=100)

    # Adds an identifier argument to the desired number of samplings
    parser.add_argument(
        '-sampling', help='Number of samplings', type=int, default=3)

    # Adds an identifier argument to the desired ReLU activation threshold
    parser.add_argument(
        '-alpha', help='ReLU activation threshold', type=float, default=0.01)

    # Adds an identifier argument to the desired Discriminator's learning rate
    parser.add_argument(
        '-d_lr', help='Discriminator learning rate', type=float, default=0.0001)

    # Adds an identifier argument to the desired Generator's learning rate
    parser.add_argument(
        '-g_lr', help='Generator learning rate', type=float, default=0.0001)

    # Adds an identifier argument to the desired batch size
    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=128)

    # Adds an identifier argument to whether normalization should be used or not
    parser.add_argument(
        '-norm', help='Dataset normalization', type=bool, default=False)

    # Adds an identifier argument to whether shuffling should be used or not
    parser.add_argument(
        '-shuffle', help='Dataset shuffling', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    input_file = args.input
    epochs = args.epochs
    output_file = args.output
    noise_dim = args.noise
    n_samplings = args.sampling
    alpha = args.alpha
    d_lr = args.d_lr
    g_lr = args.g_lr
    batch_size = args.batch_size
    norm = args.norm
    shuffle = args.shuffle

    # Loading extracted weights
    x = np.load(input_file)

    # Creating an Image Dataset
    dataset = ImageDataset(x, batch_size=batch_size, shape=(x.shape[0], x.shape[1]),
                           normalize=norm, shuffle=shuffle)

    # Creating the GAN
    gan = GAN(input_shape=(x.shape[1],), noise_dim=noise_dim,
              n_samplings=n_samplings, alpha=alpha)

    # Compiling the GAN
    gan.compile(d_optimizer=tf.optimizers.Adam(learning_rate=d_lr),
                g_optimizer=tf.optimizers.Adam(learning_rate=g_lr))

    # Fitting the GAN
    gan.fit(dataset.batches, epochs=epochs)

    # Saving GAN weights
    gan.save_weights(output_file, save_format='tf')
