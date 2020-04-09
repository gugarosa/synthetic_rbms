# ?

*This repository holds all the necessary code to run the very-same experiments described in the paper "?".*

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

## Structure

  * `data/`
    * `RSDataset`: Folder containing the RSDataset data;
    * `RSSCN7`: Folder containing the RSSCN7 data;
    * `UCMerced_LandUse`: Folder containing the UCMerced_LandUse data;
  * `models/`
    * `ensemble.py`: Ensemble-based methods, such as weight-based and majority voting;
  * `utils/`
    * `constants.py`: Constants definitions;
    * `dictionary.py`: Creates a dictionary of classes and labels;
    * `load.py`: Loads the dataset according to desired format;
    * `metrics.py`: Provides several metrics calculations;
    * `mh.py`: Wraps the meta-heuristic classes;
    * `wrapper.py`: Wraps the optimization tasks;

## How-to-Use

There are 5+1 simple steps in order to accomplish the same experiments described in the paper:

 * Install the requirements;
 * Pre-train RBMs and save their weights;
 * Pre-train GANs with RBMs-saved weights as the input data;
 * Samples new weights from a pre-trained GAN;
 * Reconstruct pre-trained RBMs with original and sampled weights;
 * (Optional) Inspect the distance between original and sampled weights.

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

After pre-training RBMs and saving their weights, we can now proceed to training a GAN with the saved weights as its input. Just run the following script and invoke their helper:

```python gan_training.py -h```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### GANs Sampling

With a pre-trained GAN in hands, it is now possible to sample new weights based on what it has learned. Therefore, run the following script in order to fulfill that purpose:

```python gan_sampling.py -h```

*Note that the saved models uses `.pth` or `tensorflow` extensions, while saved weights will use a `*.npy` file.*

### RBMs Reconstruction

Finally, with the new sampled weights, it is now possible to reconstruct the original RBMs and check if the new weights have provided some diversity. Please, run the following script:

```python rbm_reconstruction.py -h```

*Note that this script will perform two reconstructions for each pre-trained RBM: original weights and sampled weights.*

### (Optional) Weights Inspection

As an optional procedure, one can also inspect the sampled weights and calculate its euclidean distance regarding the original weights. Please use the following script in order to accomplish such an approach:

```python weights_inspection.py -h```

*Note that this script uses matplotlib as an additional package.*
