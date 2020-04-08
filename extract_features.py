import argparse

import torch

import loader as l
import utils as u


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Extracts RBM-based features.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'mnist', 'fmnist', 'kmnist'])

    # Adds an identifier argument to the desired pre-trained model path
    parser.add_argument('model', help='Input file for the pre-trained model', type=str)

    # Adds an identifier argument to the desired output file name
    parser.add_argument(
        'output', help='Output file for extracted features', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_model = args.model
    output_file = args.output

    # Loads the training data
    train, _ = l.load_dataset(name=dataset)

    # Loads the pre-trained model
    model = torch.load(input_model)

    # Performs a forward pass over the training data, i.e., feature extraction
    train_probs, _ = model.forward(train)

    # Outputting extracted features to a numpy file
    u.save_as_numpy(train_probs, output_file=output_file)
