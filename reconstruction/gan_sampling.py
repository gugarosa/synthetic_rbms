import argparse
import os

import numpy as np
import tensorflow as tf
from nalp.datasets.image import ImageDataset
from nalp.models.gan import GAN

import utils.stream as s

os.environ['TF_DETERMINISTIC_OPS'] = '1'


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Samples RBMs weights from a pre-trained GAN.')

    parser.add_argument(
        'n_samples', help='Number of weights to be sampled', type=int)

    parser.add_argument(
        'input_model', help='Input name for saved model', type=str)

    parser.add_argument(
        'input_shape', help='Input shape for saved model', type=int)

    parser.add_argument(
        '-noise', help='Noise dimension', type=int, default=10000)

    parser.add_argument(
        '-sampling', help='Number of samplings', type=int, default=3)

    parser.add_argument(
        '-alpha', help='ReLU activation threshold', type=float, default=0.01)

    parser.add_argument(
        '-d_lr', help='Discriminator learning rate', type=float, default=0.0001)

    parser.add_argument(
        '-g_lr', help='Generator learning rate', type=float, default=0.0001)

    parser.add_argument(
        '-seed', help='Tensorflow seed', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    n_samples = args.n_samples
    input_model = args.input_model
    input_shape = args.input_shape
    noise_dim = args.noise
    n_samplings = args.sampling
    alpha = args.alpha
    d_lr = args.d_lr
    g_lr = args.g_lr
    seed = args.seed

    # Setting Tensorflow and Numpy random seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Creating the GAN
    gan = GAN(input_shape=(input_shape,), noise_dim=noise_dim,
              n_samplings=n_samplings, alpha=alpha)

    # Compiling the GAN
    gan.compile(d_optimizer=tf.optimizers.Adam(learning_rate=d_lr),
                g_optimizer=tf.optimizers.Adam(learning_rate=g_lr))

    # Loading GAN weights
    gan.load_weights(f'models/{input_model}')

    # Creating a noise tensor for further sampling
    z = tf.random.normal([n_samples, 1, 1, noise_dim])

    # Sampling artificial weights
    sampled_weights = tf.reshape(gan.G(z), (n_samples, input_shape))

    # Outputting sampled weights to a numpy file
    s.save_tf_as_numpy(sampled_weights, output_file=f'weights/{input_model}')
