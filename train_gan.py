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
        usage='Trains a GAN using RBM-extracted feature vectors.')

    # Adds an identifier argument to the desired extracted features
    parser.add_argument(
        'input', help='Input file for the extracted features', type=str)

    # Adds an identifier argument to the desired output model file
    parser.add_argument(
        'output', help='Output file for saved model', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    input_file = args.input
    output_file = args.output

    # Loading extracted features
    x = np.load(input_file)

    # Creating an Image Dataset
    dataset = ImageDataset(x, batch_size=128, shape=(
        x.shape[0], x.shape[1]), normalize=False)

    # Creating the GAN
    gan = GAN(input_shape=(x.shape[1],),
              noise_dim=100, n_samplings=3, alpha=0.01)

    # Compiling the GAN
    gan.compile(d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                g_optimizer=tf.optimizers.Adam(learning_rate=0.0001))

    # Fitting the GAN
    gan.fit(dataset.batches, epochs=10)

    # Saving GAN weights
    gan.save_weights(output_file, save_format='tf')
