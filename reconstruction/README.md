## Structure

  * `customized_nalp/`: A customized version of the NALP library in regard to this experiment;
  * `models/`: Folder for saving the output models, such as `.pth` and `tensorflow` ones;
  * `utils/`
    * `stream.py`: Common loading and saving methods;
  * `weights/`: Folder for saving the output weights, which will use `.npy` extensions.

## How-to-Use

There are 4 simple steps in order to accomplish the same experiments described in the paper:

 * Install the requirements;
 * Pre-train RBMs and save their weights;
 * Pre-train and samples data from GANs with RBMs-saved weights as the input data;
 * Performs a comparison search between each GAN epoch's sampled weights and the original RBM weights;

Additionally, you can perform the whole experimentation step with the provided shell script, as follows:

```./create_optimized_synthetic_rbms.sh```
 
### Installation

Please install all the pre-needed requirements using:

```pip install -r requirements.txt```

### RBMs Pre-Training

Our first script helps you in pre-training an RBM and saving its weights. With that in mind, just run the following script with the input arguments:

```python rbm_training.py -h```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### GANs Pre-Training and Sampling

After pre-training RBMs and saving their weights, we can now proceed in training a GAN with the saved weights as its input. Just run the following script and invoke its helper:

```python gan_training_and_sampling.py -h```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### Finding GAN's Best Sampled Weight Guided by RBM Reconstruction

With a pre-trained and sampled weights from the GAN in hands, it is now possible to reconstruct these weights over a validation set and compare at what epoch the GAN could outperform the original RBM. Therefore, run the following script in order to fulfill that purpose:

```python find_best_sampled_weight.py -h```

*Note that the saved models uses `.pth` or `tensorflow` extensions, while saved weights will use a `*.npy` file.*