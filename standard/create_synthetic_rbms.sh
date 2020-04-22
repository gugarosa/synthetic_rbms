# Defining a counter for iterating purposes
i=0

# Defining a constant for number of pre-trained RBMs, i.e., input samples for the GAN
N_RBMS=128

# Defining a constant to hold the dataset
DATASET="mnist"

# Defining a constant to hold RBMs related output names
RBM_PATH="vanilla_rbm"

# Defining a constant to hold GANs related output names
GAN_PATH="vanilla_gan"

# Creating a loop of `N_RBMS`
while [ $i -lt $N_RBMS ]; do
    # Pre-training amount of desired RBMs
    python rbm_training.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} -epochs 5

    # Incrementing the counter
    i=$(($i+1))
done

# Pre-training GAN with RBMs weights
python gan_training.py ${N_RBMS} ${RBM_PATH} ${GAN_PATH} -batch_size 4 -noise 10000 -epochs 1000

# Sampling new weights from pre-trained GAN
python gan_sampling.py ${GAN_PATH} ${GAN_PATH} -noise 10000

# Restting the counter for iterating purposes
i=0

# Creating a loop of `N_RBMS`
while [ $i -lt $N_RBMS ]; do
    # Reconstructing all pre-trained RBMs with original weights
    python rbm_reconstruction.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} ${GAN_PATH} -alpha 0

    # Reconstructing all pre-trained RBMs with sampled weights
    python rbm_reconstruction.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} ${GAN_PATH} -alpha 1

    # Reconstructing all pre-trained RBMs with linear combination of original and sampled weights
    python rbm_reconstruction.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} ${GAN_PATH} -alpha 0.01

    # Incrementing the counter
    i=$(($i+1))
done

