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

# Defining a constant to hold the sampled weight epoch to be used
SAMPLED_EPOCH=1040

# Creating a loop of `N_RBMS`
while [ $i -lt $N_RBMS ]; do
    # Classifying the test set with all pre-trained RBMs and original weights
    python rbm_classification.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} ${GAN_PATH} ${SAMPLED_EPOCH} -alpha 0

    # Classifying the test set with all pre-trained RBMs and sampled weights
    python rbm_classification.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} ${GAN_PATH} ${SAMPLED_EPOCH} -alpha 1

    # Classifying the test set with all pre-trained RBMs using a linear combination of original and sampled weights
    python rbm_classification.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} ${GAN_PATH} ${SAMPLED_EPOCH} -alpha 0.9

    # Incrementing the counter
    i=$(($i+1))
done
