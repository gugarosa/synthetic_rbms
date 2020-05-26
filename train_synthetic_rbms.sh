# Defining a counter for iterating purposes
i=0

# Defining a constant for number of pre-trained RBMs, i.e., input samples for the GAN
N_RBMS=32

# Defining a constant to hold the dataset
DATASET="mnist"

# Defining a constant to hold RBMs related output names
RBM_PATH="vanilla_rbm"

# Creating a loop of `N_RBMS`
while [ $i -lt $N_RBMS ]; do
    # Pre-training amount of desired RBMs
    python rbm_training.py ${DATASET} ${RBM_PATH}_${i} ${RBM_PATH}_${i} -seed ${i}

    # Incrementing the counter
    i=$(($i+1))
done