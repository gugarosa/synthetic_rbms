import argparse

import tensorflow as tf
from nalp.models.gan import GAN

import utils.saver as s


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Samples new weights vectors from a pre-trained GAN')

    # Adds an identifier argument to the desired pre-trained model path
    parser.add_argument(
        'input', help='Input file for the pre-trained GAN', type=str)

    # Adds an identifier argument to the desired output file name
    parser.add_argument(
        'output', help='Output file for sampled weights', type=str)

    # Adds an identifier argument to the desired size of generated samples
    parser.add_argument(
        '-size', help='Amount of generated samples', type=int, default=784)

    # Adds an identifier argument to the desired number of features
    parser.add_argument(
        '-n_features', help='Amount of features of pre-trained GAN', type=int, default=256)

    # Adds an identifier argument to the desired noise dimension
    parser.add_argument(
        '-noise', help='Noise dimension', type=int, default=100)

    # Adds an identifier argument to the desired number of samplings
    parser.add_argument(
        '-sampling', help='Number of samplings', type=int, default=3)

    # Adds an identifier argument to the desired ReLU activation threshold
    parser.add_argument(
        '-alpha', help='ReLU activation threshold', type=float, default=0.01)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    input_file = args.input
    output_file = args.output
    size = args.size
    n_features = args.n_features
    noise_dim = args.noise
    n_samplings = args.sampling
    alpha = args.alpha

    # Creating the GAN
    gan = GAN(input_shape=(n_features,), noise_dim=noise_dim,
              n_samplings=n_samplings, alpha=alpha)

    # Loading pre-trained GAN weights
    gan.load_weights(input_file).expect_partial()

    # Creating a noise tensor for further sampling
    z = tf.random.normal([size, 1, 1, noise_dim])

    # Sampling from GAN
    sampled_z = gan.G(z)

    # Outputting sampled weights to a numpy file
    s.save_tf_as_numpy(sampled_z, output_file=output_file)
