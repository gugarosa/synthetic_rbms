# Adversarially Generated Restricted Boltzmann Machines

*This repository holds all the necessary code to run the very-same experiments described in the paper "Adversarially Generated Restricted Boltzmann Machines".*

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

## Structure

  * `libraries/`: Folder containing a customized version of the NALP library in regard to this experiment;
  * `models/`: Folder for saving the output models, such as `.pth` and `tensorflow` ones;
  * `utils/`
    * `stream.py`: Common loading and saving methods;
  * `weights/`: Folder for saving the output weights, which will use `.npy` extensions.

## How-to-Use

There are 4 simple steps in order to accomplish the same experiments described in the paper:

 * Install the requirements;
 * Pre-train RBMs and save their weights;
 * Pre-train GANs with RBMs-saved weights as the input data and validate its performance;
 * Perform the final evaluation comparison original and sampled weights in the testing set;

*Pre-trained models are also available at: http://recogna.tech/files*.
 
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

If you wish to sample any additional weights, please use:

```python gan_sampling.py -h```

### Finding GAN's Best Sampled Weight Guided by RBM Reconstruction or Classification

With a pre-trained and sampled weights from the GAN in hands, it is now possible to reconstruct / classify these weights over a validation set and compare at what epoch the GAN could outperform the original RBM. Therefore, run the following script in order to fulfill that purpose:

```python find_best_sampled_weight_rec.py -h``` or ```python find_best_sampled_weight_clf.py -h```

*Note that the saved models uses `.pth` or `tensorflow` extensions, while saved weights will use a `*.npy` file.*

### Final Evaluation

After finding the best GAN's sampled weight, it is now possible to perform a final reconstruction / classification over the testing set. To accomplish such a procedure, please use:

```python rbm_reconstruction.py -h``` or ```python rbm_classification.py -h```