# ?

*This repository holds all the necessary code to run the very-same experiments described in the paper "?".*

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

## Structure

  * `models/`: Folder for saving the output models, such as `.pth` and `tensorflow` ones;
  * `utils/`
    * `stream.py`: Common loading and saving methods;
  * `weights/`: Folder for saving the output weights, which will use `.npy` extensions.

## How-to-Use

There are 5+1 simple steps in order to accomplish the same experiments described in the paper:

 * Install the requirements;
 * Pre-train RBMs and save their weights;
 * Pre-train GANs with RBMs-saved weights as the input data;
 * Samples new weights from a pre-trained GAN;
 * Reconstruct pre-trained RBMs with original and sampled weights;
 * (Optional) Inspect the mosaic between original and sampled weights.

Additionally, you can perform the whole experimentation step with the provided shell script, as follows:

```./create_synthetic_rbms.sh```
 
### Installation

Please install all the pre-needed requirements using:

```pip install -r requirements.txt```

### RBMs Pre-Training

Our first script helps you in pre-training an RBM and saving its weights. With that in mind, just run the following script with the input arguments:

```python rbm_training.py -h```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### GANs Pre-Training

After pre-training RBMs and saving their weights, we can now proceed in training a GAN with the saved weights as its input. Just run the following script and invoke its helper:

```python gan_training.py -h```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### GANs Sampling

With a pre-trained GAN in hands, it is now possible to sample new weights based on what it has learned. Therefore, run the following script in order to fulfill that purpose:

```python gan_sampling.py -h```

*Note that the saved models uses `.pth` or `tensorflow` extensions, while saved weights will use a `*.npy` file.*

### RBMs Reconstruction

Finally, with the new sampled weights, it is now possible to reconstruct the original RBMs and check if the new weights have provided some diversity. Please, run the following script:

```python rbm_reconstruction.py -h```

*Note that this script uses a parameter for reconstructing weights: 0 will use original weights, 1 will use sampled weights, and any value between 0 and 1 will use a linear combination between both.*

### (Optional) Weights Inspection

As an optional procedure, one can also inspect mosaics from the sampled weights and the original weights. Please use the following script in order to accomplish such an approach:

```python weights_inspection.py -h```

*Note that this script uses learnergy to create the mosaics.*
