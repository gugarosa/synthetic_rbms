import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Embedding,
                                     MaxPool1D)

import nalp.utils.logging as l
from nalp.core.model import Discriminator

logger = l.get_logger(__name__)


class EmbeddedTextDiscriminator(Discriminator):
    """A EmbeddedTextDiscriminator class stands for the text-discriminative part of a Generative Adversarial Network.

    """

    def __init__(self, vocab_size=1, max_length=1, embedding_size=32, n_filters=[64], filters_size=[1], dropout_rate=0.25):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            max_length (int): Maximum length of the sequences.
            embedding_size (int): The size of the embedding layer.
            n_filters (list): Number of filters to be applied.
            filters_size (list): Size of filters to be applied.
            dropout_rate (float): Dropout activation rate.

        """

        logger.info(
            'Overriding class: Discriminator -> EmbeddedTextDiscriminator.')

        # Overrides its parent class with any custom arguments if needed
        super(EmbeddedTextDiscriminator, self).__init__(name='D_text')

        # Creates an embedding layer
        self.embedding = Embedding(
            vocab_size, embedding_size, name='embedding')

        # Defining a list for holding the convolutional layers
        self.conv = [Conv2D(n, (k, embedding_size), strides=(
            1, 1), padding='valid', name=f'conv_{k}') for n, k in zip(n_filters, filters_size)]

        # Defining a list for holding the pooling layers
        self.pool = [MaxPool1D(max_length - k + 1, 1, name=f'pool_{k}')
                     for k in filters_size]

        # Defining a linear layer for serving as the `highway`
        self.highway = Dense(sum(n_filters), name='highway')

        # Defining the dropout layer
        self.drop = Dropout(dropout_rate, name='drop')

        # And finally, defining the output layer
        self.out = Dense(2, name='out')

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # Passing down the embedding layer
        x = self.embedding(x)

        # Expanding the last dimension
        x = tf.expand_dims(x, -1)

        # Passing down the convolutional layers following a ReLU activation
        # and removal of third dimension
        convs = [tf.squeeze(tf.nn.relu(conv(x)), 2) for conv in self.conv]

        # Passing down the pooling layers per convolutional layer
        pools = [pool(conv) for pool, conv in zip(self.pool, convs)]

        # Concatenating all the pooling outputs into a single tensor
        x = tf.concat(pools, 2)

        # Calculating the output of the linear layer
        hw = self.highway(x)

        # Calculating the `highway` layer
        x = tf.math.sigmoid(hw) * tf.nn.relu(hw) + (1 - tf.math.sigmoid(hw)) * x

        # Calculating the output with a dropout regularization
        x = self.out(self.drop(x, training=training))

        return x
