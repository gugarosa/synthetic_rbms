import argparse

import torch
from learnergy.models.rbm import RBM

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Trains an RBM.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'mnist', 'fmnist', 'kmnist'])

    # Adds an identifier argument to the desired number of training epochs
    parser.add_argument('epochs', help='Number of training epochs', type=int)

    # Adds an identifier argument to the desired output model file
    parser.add_argument(
        'output', help='Output file for saved model', type=str)

    # Adds an identifier argument to the desired number of visible units
    parser.add_argument(
        '-n_visible', help='Number of visible units', type=int, default=784)

    # Adds an identifier argument to the desired number of hidden units
    parser.add_argument(
        '-n_hidden', help='Number of hidden units', type=int, default=256)

    # Adds an identifier argument to the desired number of CD steps
    parser.add_argument(
        '-steps', help='Number of CD steps', type=int, default=1)

    # Adds an identifier argument to the desired learning rate
    parser.add_argument(
        '-lr', help='Learning rate', type=float, default=0.1)

    # Adds an identifier argument to the desired momentum
    parser.add_argument(
        '-momentum', help='Momentum', type=float, default=0)

    # Adds an identifier argument to the desired weight decay
    parser.add_argument(
        '-decay', help='Weight decay', type=float, default=0)

    # Adds an identifier argument to the desired temperature
    parser.add_argument(
        '-temp', help='Temperature', type=float, default=1)

    # Adds an identifier argument to whether GPU should be used or not
    parser.add_argument(
        '-gpu', help='GPU usage', type=bool, default=True)

    # Adds an identifier argument to the desired batch size
    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=128)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    epochs = args.epochs
    output_file = args.output
    n_visible = args.n_visible
    n_hidden = args.n_hidden
    steps = args.steps
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    T = args.temp
    gpu = args.gpu
    batch_size = args.batch_size

    # Loads the training data
    train, _ = l.load_dataset(name=dataset)

    # Creates an RBM
    model = RBM(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                momentum=momentum, decay=decay, temperature=T, use_gpu=gpu)

    # Fits an RBM
    mse, pl = model.fit(train, batch_size=batch_size, epochs=epochs)

    # Saves the RBM
    torch.save(model, output_file)
