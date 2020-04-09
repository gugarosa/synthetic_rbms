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
        usage='Trains a GAN using RBMs weights.')

    parser.add_argument(
        'runs', help='Number of RBMs used for weights extraction', type=int)

    parser.add_argument(
        'input_weight', help='Input name for the extracted weights without folder, index and extension', type=str)

    parser.add_argument(
        'output_model', help='Output name for saved model', type=str)

    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=4)

    parser.add_argument(
        '-norm', help='Dataset normalization', type=bool, default=False)

    parser.add_argument(
        '-shuffle', help='Dataset shuffling', type=bool, default=False)

    parser.add_argument(
        '-noise', help='Noise dimension', type=int, default=100)

    parser.add_argument(
        '-sampling', help='Number of samplings', type=int, default=3)

    parser.add_argument(
        '-alpha', help='ReLU activation threshold', type=float, default=0.01)

    parser.add_argument(
        '-d_lr', help='Discriminator learning rate', type=float, default=0.0001)

    parser.add_argument(
        '-g_lr', help='Generator learning rate', type=float, default=0.0001)

    parser.add_argument(
        '-epochs', help='Number of training epochs', type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    runs = args.runs
    input_weight = args.input_weight
    output_model = args.output_model
    batch_size = args.batch_size
    norm = args.norm
    shuffle = args.shuffle
    noise_dim = args.noise
    n_samplings = args.sampling
    alpha = args.alpha
    d_lr = args.d_lr
    g_lr = args.g_lr
    epochs = args.epochs
    
    # Creating an empty array for holding input data
    x = np.zeros([1, 0])

    for i in range(runs):
        # Loading RBM weights
        W = np.load(f'weights/{input_weight}_{i}.npy')

        # Flatenning weights
        W = np.reshape(W, [1, -1])

        # Concatenating to input array
        x = np.hstack([x, W])

    # Reshaping input back to (n_runs, n_features)
    x = np.reshape(x, [runs, -1])

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
    gan.save_weights(f'models/{output_model}', save_format='tf')
