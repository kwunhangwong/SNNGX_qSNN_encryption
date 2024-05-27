# SNNGX_qSNN_encryption

![commit badge](https://img.shields.io/badge/private-8A2BE2)
![commit badge](https://img.shields.io/badge/Binary-Neural%20Network-blue)

## Description

**This project performs efficient SNNGX weight-bit (MSB) encryption against quantized Spiking neural networks as an IP protection method and demonstrates biologically-plausible UNIVERSAL UTILITY on SNN models** 

**There are SNNGX Encryption implementation, Random bit Encrpytioin baseline and Potential attack threats in this repository and some pretrained weights over NMNIST and DVS128Gesture.**

![FGSM_GA_illustration_v2](https://github.com/u3556440/SNN_security/assets/56315946/976bca55-fb5a-4f1a-bd0e-18a3aa95ad84)

**This project was supported by the University of Hong Kong, EEE. \
Currently restricted to internal usage only.**

## Quick Start

pytorch>=1.10.0+cu111  
tonic>=1.4.3 

```
git clone https://github.com/u3556440/SNNGX_qSNN_encryption.git
```


## Usage

*Dataset is not included in our repository. \
Please indicate your path to the Dataset (NMNIST/MNIST) in the below bash commands.*


1. SNNGX Encryption (Genetic Algorithm):

```
cd Encryption_SNNGX
```
```
python3 __main__.py [-b] [--epsil] [--name] [--subset] [--mutate] [--gen] [--Dataset] [--Dpath]
```

Flag | Metavar | Help
--- | --- | --- 
`-b`/`--batch` | batch_size | Batch size for dataset, model, GA-subset
`--epsil` | GA_epsilon | Final number of bits to be converged by GA
`--name` | Documenting_fitness | Recording the best fitness function per generation 
`--subset` | number_of_subset_images | No. of Samples for calculating fitness loss function
`--mutate` | GA_mutate_chance | pixel-wise mutation probability (p=0.05) 
`--gen` | GA_generations | # of GA_generations
`--Dataset` | Dataset_Target | Available Dataset: NMNIST/MNIST
`--Dpath` | Dataset_Path | Please input your local path to the dataset

You may wish to modify the follwing # commented code in __main__.py to change the search space 
from all layer to single target layer, and vice versa.

![searchSpace_GA](https://github.com/u3556440/SNN_security/assets/56315946/75ded59a-1b0e-4cc4-b63f-4ccce4139782)

2. Random bit Encryption:

```
cd Encryption_Random_bit
```
```
python3 __main__.py [-b] [-nb] [-nt] [--Dataset] [--Dpath]
```

Flag | Metavar | Help
--- | --- | --- 
`-b`/`--batch` | batch_size | Batch size for dataset and model
`-nb`/`--numbit` | number_of_flipped_BITS | # of Random BITS flipped
`-nt`/`--numtrial` | number_of_random_trials | # of trial repetition
`--Dataset` | Dataset_Target | Available Dataset: "NMNIST"/"DVS128_Gesture"
`--Dpath` | Dataset_Path | Please input your local path to the dataset

You may wish to modify the follwing # commented code in __main__.py to change the search space 
from all layer to single target layer, and vice versa.

![searchSpace_random_flipping](https://github.com/u3556440/SNN_security/assets/56315946/bead64b1-8743-4b46-930f-82a63cfdfbd3)