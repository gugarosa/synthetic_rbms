import argparse

import numpy as np
import torch
from sklearn.svm import SVC

import utils.stream as s


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Classifies an RBM with linear combination of original and sampled weights.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'mnist', 'fmnist', 'kmnist'])

    # Adds an identifier argument to the desired pre-trained model path
    parser.add_argument(
        'input_model', help='Input name for the pre-trained RBM', type=str)

    # Adds an identifier argument to the desired sampled weights file
    parser.add_argument(
        'input_sampled', help='Input name for the sampled weight file', type=str)

    # Adds an identifier argument to the desired  file
    parser.add_argument(
        '-alpha', help='Constant used to calculate the linear combination between weights', type=float, default=0.01)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_model = args.input_model
    input_sampled = args.input_sampled
    alpha = args.alpha

    # Instantiates an SVM
    clf = SVC(gamma='auto')

    # Loads the training and testing data
    train, _, test = s.load_dataset(name=dataset)

    # Transforming datasets into tensors
    x_train, y_train = s.dataset_as_tensor(train)
    x_test, y_test = s.dataset_as_tensor(test)

    # Reshaping tensors
    x_train = x_train.view(len(train), 784)
    x_test = x_test.view(len(test), 784)

    # Loads the pre-trained model
    model = torch.load(f'models/{input_model}.pth')

    # Loading original and sampled weights
    W_sampled = np.load(f'weights/{input_sampled}.npy')

    # Reshaping weights to correct dimension
    W_sampled = np.reshape(W_sampled, [W_sampled.shape[0], model.n_visible, model.n_hidden])

    # Resetting biases for fair comparison
    model.a = torch.nn.Parameter(torch.zeros(model.n_visible))
    model.b = torch.nn.Parameter(torch.zeros(model.n_hidden))

    # For every sampled weight
    for i in range(W_sampled.shape[0]):
        # Applying linear combination of original and sampled weights as new weights
        model.W = torch.nn.Parameter((1 - alpha) * model.W + alpha * torch.from_numpy(W_sampled[i]).to(model.device))

        # Checking model device type
        if model.device == 'cuda':
            # Applying its parameters as cuda again
            model = model.cuda()
            x_train = x_train.cuda()
            x_test = x_test.cuda()

        # Extract features from the original RBM
        f_train = model.forward(x_train)
        f_test = model.forward(x_test)

        # Instantiates an SVM
        clf = SVC(gamma='auto')

        # Fits a classifier
        clf.fit(f_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

        # Performs the final classification
        acc = clf.score(f_test.detach().cpu().numpy(), y_test.detach().cpu().numpy())

        print(acc)