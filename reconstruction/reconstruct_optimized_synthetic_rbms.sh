# Defining a counter for iterating purposes
i=0

# Defining a constant for number of pre-trained RBMs, i.e., input samples for the GAN
N_RBMS=64

# Defining a constant to hold the dataset
DATASET="mnist"

# Defining a constant to hold RBMs related output names
RBM_PATH="vanilla_rbm"

# Defining a constant to hold GANs related output names
GAN_PATH="vanilla_gan"

# Creating a loop of `N_RBMS`
while [ $i -lt $N_RBMS ]; do
    # Reconstructing the test set with all pre-trained RBMs and original weights
    python rbm_reconstruction.py ${DATASET} ${RBM_PATH}_${i} ${GAN_PATH} -alpha 0

    # Reconstructing the test set with all pre-trained RBMs and sampled weights
    python rbm_reconstruction.py ${DATASET} ${RBM_PATH}_${i} ${GAN_PATH} -alpha 1

    # Reconstructing the test set with all pre-trained RBMs using a linear combination of original and sampled weights
    python rbm_reconstruction.py ${DATASET} ${RBM_PATH}_${i} ${GAN_PATH} -alpha 0.9

    # Incrementing the counter
    i=$(($i+1))
done
