import argparse

import torch

import utils.loader as l
import utils.saver as s


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Extracts RBM-based weight vectors.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'mnist', 'fmnist', 'kmnist'])

    # Adds an identifier argument to the desired pre-trained model path
    parser.add_argument('input', help='Input file for the pre-trained RBM', type=str)

    # Adds an identifier argument to the desired output file name
    parser.add_argument(
        'output', help='Output file for extracted weights', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_file = args.input
    output_file = args.output

    # Loads the training data
    train, _ = l.load_dataset(name=dataset)

    # Loads the pre-trained model
    model = torch.load(input_file)

    # Outputting extracted weights to a numpy file
    s.save_torch_as_numpy(model.W, output_file=output_file)
