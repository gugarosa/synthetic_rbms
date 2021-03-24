# Fast Ensemble Learning Using Adversarially-Generated Restricted Boltzmann Machines

*This repository holds all the necessary code to run the very-same experiments described in the paper "Fast Ensemble Learning Using Adversarially-Generated Restricted Boltzmann Machines".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
@misc{rosa2021fast,
      title={Fast Ensemble Learning Using Adversarially-Generated Restricted Boltzmann Machines}, 
      author={Gustavo H. de Rosa and Mateus Roder and João P. Papa},
      year={2021},
      eprint={2101.01042},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

---

## Structure

  * `libraries/`: Folder containing a customized version of the NALP library regarding this experiment;
  * `models/`: Folder for saving the output models, such as `.pth` and `tensorflow` ones;
  * `utils/`
    * `stream.py`: Common loading and saving methods;
  * `weights/`: Folder for saving the output weights, which will use `.npy` extensions.

---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

---

## Usage

### RBMs Pre-Training

Our first script helps you in pre-training an RBM and saving its weights. With that in mind, just run the following script with the input arguments:

```Python
python rbm_training.py -h
```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### GANs Pre-Training and Sampling

After pre-training RBMs and saving their weights, we can now train a GAN with the saved weights as its input. Just run the following script and invoke its helper:

```Python
python gan_training_and_sampling.py -h
```

If you wish to sample any additional weights, please use:

```Python
python gan_sampling.py -h
```

### Finding GAN's Best Sampled Weight Guided by RBM Reconstruction or Classification

With a pre-trained and sampled weights from the GAN in hands, it is now possible to reconstruct/classify these weights over a validation set and compare at what epoch the GAN could outperform the original RBM. Therefore, run the following script in order to fulfill that purpose:

```Python
python find_best_sampled_weight_rec.py -h
```

or

```Python
python find_best_sampled_weight_clf.py -h
```

*Note that the saved models use `.pth` or `tensorflow` extensions, while saved weights will use a `*.npy` file.*

### Final Evaluation

After finding the best GAN's sampled weight, it is now possible to perform a final reconstruction/classification over the testing set. To accomplish such a procedure, please use:

```Python
python rbm_reconstruction.py -h
```

or

```Python
python rbm_classification.py -h
```

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell scripts, as follows:

```Bash
./train_synthetic_rbms.sh
```

```Bash
./validate_synthetic_rbms.sh
```

```Bash
./test_synthetic_rbms.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
