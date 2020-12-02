import argparse

import torch
from learnergy.models.bernoulli import RBM

import utils.stream as s


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Trains an RBM and saves its weights.')

    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'mnist', 'fmnist', 'kmnist'])

    parser.add_argument(
        'output_model', help='Output name for saved model', type=str)

    parser.add_argument(
        'output_weight', help='Output name for saved weight', type=str)

    parser.add_argument(
        '-split', help='Percentage of training data to be converted into validation data', type=float, default=0.2)

    parser.add_argument(
        '-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument(
        '-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument(
        '-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument(
        '-lr', help='Learning rate', type=float, default=0.1)

    parser.add_argument(
        '-momentum', help='Momentum', type=float, default=0)

    parser.add_argument(
        '-decay', help='Weight decay', type=float, default=0)

    parser.add_argument(
        '-temp', help='Temperature', type=float, default=1)

    parser.add_argument(
        '-gpu', help='GPU usage', type=bool, default=True)

    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument(
        '-epochs', help='Number of training epochs', type=int, default=10)

    parser.add_argument(
        '-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    output_model = args.output_model
    output_weight = args.output_weight
    split = args.split
    n_visible = args.n_visible
    n_hidden = args.n_hidden
    steps = args.steps
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    T = args.temp
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    seed = args.seed

    # Loads the training data
    train, _, _ = s.load_dataset(name=dataset, val_split=split)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Creates an RBM
    model = RBM(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                momentum=momentum, decay=decay, temperature=T, use_gpu=True)

    # Fits an RBM
    mse, pl = model.fit(train, batch_size=batch_size, epochs=epochs)

    # Saves the RBM
    torch.save(model, f'models/{output_model}.pth')

    # Outputting extracted weights to a numpy file
    s.save_torch_as_numpy(model.W, output_file=f'weights/{output_weight}.npy')
