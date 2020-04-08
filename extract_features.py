import torch
from learnergy.models.rbm import RBM

import loader as l
import utils as u

# Loads the training data
train, _ = l.load_dataset(name='mnist')

# Creates an RBM
model = RBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
            momentum=0, decay=0, temperature=1, use_gpu=True)

# Fits an RBM
mse, pl = model.fit(train, batch_size=128, epochs=1)

# Reconstructs the training data, i.e., feature extraction
_, rec_train = model.reconstruct(train)

# Outputting extracted features to a numpy file
u.save_as_numpy(rec_train, output_file='out.npy')
