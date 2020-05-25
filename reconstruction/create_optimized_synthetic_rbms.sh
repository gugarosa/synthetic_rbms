# Defining a counter for iterating purposes
i=0

# Defining a constant for number of pre-trained RBMs, i.e., input samples for the GAN
N_RBMS=32

# Defining a constant to hold the dataset
DATASET="mnist"

# Defining a constant to hold RBMs related output names
RBM_PATH="vanilla_rbm"

# Defining a constant to hold GANs related output names
GAN_PATH="vanilla_gan"

# Creating a loop of `N_RBMS`
while [ $i -lt $N_RBMS ]; do
    # Pre-training amount of desired RBMs
    python rbm_training.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} -seed ${i}

    # Incrementing the counter
    i=$(($i+1))
done

# Pre-training GAN with RBMs weights
python gan_training_and_sampling.py ${N_RBMS} ${RBM_PATH} ${GAN_PATH} -batch_size 2

# Resetting the counter for iterating purposes
i=0

# Creating a loop of `N_RBMS`
while [ $i -lt $N_RBMS ]; do
    # Finding the best sampled weight by reconstructing an RBM over a validation set with original and sampled weights
    python find_best_sampled_weight.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} ${GAN_PATH}

    # Incrementing the counter
    i=$(($i+1))
done