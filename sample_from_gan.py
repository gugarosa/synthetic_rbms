import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from nalp.models.gan import GAN


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Samples new feature vectors from a pre-trained GAN')

    # Adds an identifier argument to the desired pre-trained model path
    parser.add_argument(
        'input', help='Input file for the pre-trained GAN', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    input_file = args.input

    # Creating the GAN
    gan = GAN(input_shape=(256,), noise_dim=100, n_samplings=3, alpha=0.01)

    # Loading pre-trained GAN weights
    gan.load_weights(input_file).expect_partial()

    # Creating a noise tensor for further sampling
    z = tf.random.normal([16, 1, 1, 100])

    # Sampling an artificial image
    sampled_images = tf.reshape(gan.G(z), (16, 16, 16))

    # Creating a pyplot figure
    fig = plt.figure(figsize=(4,4))

    # For every possible generated image
    for i in range(sampled_images.shape[0]):
        # Defines the subplot
        plt.subplot(4, 4, i+1)

        # Plots the image to the figure
        plt.imshow(sampled_images[i, :, :] * 127.5 + 127.5, cmap='gray')

        # Disabling the axis
        plt.axis('off')

    # Showing the plot
    plt.show()