import argparse

import numpy as np
import torch
from sklearn.svm import SVC

import utils.stream as s
import learnergy.utils.logging as l

logger = l.get_logger(__name__)


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Finds the best sampled weight by classifying an RBM over a validation set with original and sampled weights.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'mnist', 'fmnist', 'kmnist'])

    # Adds an identifier argument to the desired pre-trained model path
    parser.add_argument(
        'input_model', help='Input name for the pre-trained RBM', type=str)

    # Adds an identifier argument to the desired weights file
    parser.add_argument(
        'input_weight', help='Input name for the weight file', type=str)

    # Adds an identifier argument to the desired sampled weights file
    parser.add_argument(
        'input_sampled', help='Input name for the sampled weight file', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_model = args.input_model
    input_weight = args.input_weight
    input_sampled = args.input_sampled

    # Instantiates an SVM
    clf = SVC(gamma='auto')

    # Loads the training and validation data
    train, val, _ = s.load_dataset(name=dataset)

    # Transforming datasets into tensors
    x_train, y_train = s.dataset_as_tensor(train)
    x_val, y_val = s.dataset_as_tensor(val)

    # Reshaping tensors
    x_train = x_train.view(len(train), 784)
    x_val = x_val.view(len(val), 784)

    # Loads the pre-trained model
    model = torch.load(f'models/{input_model}.pth')

    # Loading original and sampled weights
    W = np.load(f'weights/{input_weight}.npy')
    W_sampled = np.load(f'weights/{input_sampled}.npy')

    # Reshaping weights to correct dimension
    W = np.reshape(W, [model.n_visible, model.n_hidden])
    W_sampled = np.reshape(W_sampled, [W_sampled.shape[0], model.n_visible, model.n_hidden])

    # Resetting biases for fair comparison
    model.a = torch.nn.Parameter(torch.zeros(model.n_visible))
    model.b = torch.nn.Parameter(torch.zeros(model.n_hidden))

    # Applying original weights
    model.W = torch.nn.Parameter(torch.from_numpy(W))

    # Checking model device type
    if model.device == 'cuda':
        # Applying its parameters as cuda again
        model = model.cuda()
        x_train = x_train.cuda()
        x_val = x_val.cuda()

    # Extract features from the original RBM
    f_train = model.forward(x_train)
    f_val = model.forward(x_val)

    # Instantiates an SVM
    clf = SVC(gamma='auto')

    # Fits a classifier
    clf.fit(f_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

    # Validates the classifier
    original_acc = clf.score(f_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())

    logger.info(original_acc)

    # Defining best sampled accuracy and best epoch as zero
    best_sampled_acc, best_epoch = 0, 0

    # Iterating over all possible epochs
    for e in range(W_sampled.shape[0]):
        print(f'Weights from GAN epoch {e+1}/{W_sampled.shape[0]}')

        # Resetting biases for fair comparison
        model.a = torch.nn.Parameter(torch.zeros(model.n_visible))
        model.b = torch.nn.Parameter(torch.zeros(model.n_hidden))

        # Applying sampled weights
        model.W = torch.nn.Parameter(torch.from_numpy(W_sampled[e]))

        # Checking model device type
        if model.device == 'cuda':
            # Applying its parameters as cuda again
            model = model.cuda()
            x_train = x_train.cuda()
            x_val = x_val.cuda()

        # Extract features from the original RBM
        f_train = model.forward(x_train)
        f_val = model.forward(x_val)

        # Fits a classifier
        clf.fit(f_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

        # Validates the classifier
        sampled_acc = clf.score(f_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())

        logger.info(sampled_acc)

        # Checking if current sampled accuracy was better than previous one
        if sampled_acc < best_sampled_acc:
            # Saving best accuracy and best epoch values
            best_sampled_acc, best_epoch = sampled_acc, e

    print(f'Validation finished and best RBM found.')
    print(f'Original Accuracy: {original_acc} | Best Sampled Accuracy: {best_sampled_acc} | Epoch: {best_epoch+1}')
