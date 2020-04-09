# Defining a counter for iterating purposes
i=0

# Defining a constant for number of weights to extract from the RBMs 
N_WEIGHTS=2

# Defining a constant to hold the dataset
DATASET="mnist"

# Defining a constant to hold RBMs related output names
RBM_PATH="vanilla_rbm"

# Defining a constant to hold GANs related output names
GAN_PATH="vanilla_gan"

# Creating a loop of `N_WEIGHTS`
while [ $i -lt $N_WEIGHTS ]; do
    # Pre-training amount of desired RBMs
    # python rbm_training.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} -epochs 1

    # Incrementing the counter
    i=$(($i+1))
done

# Pre-training GAN with RBMs weights
# python gan_training.py ${N_WEIGHTS} ${RBM_PATH} ${GAN_PATH}

# Sampling new weights from pre-trained GAN
# python gan_sampling.py ${GAN_PATH} ${GAN_PATH}

# Restting the counter for iterating purposes
i=0

# Creating a loop of `N_WEIGHTS`
while [ $i -lt $N_WEIGHTS ]; do
    # Reconstructing all pre-trained RBMs with pre-trained weights
    python rbm_reconstruction.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i}

    # Reconstructing all pre-trained RBMs with sampled weights
    python rbm_reconstruction.py ${DATASET} ${RBM_PATH}_${i} ${GAN_PATH}

    # Incrementing the counter
    i=$(($i+1))
done

