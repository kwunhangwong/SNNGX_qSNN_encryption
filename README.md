# SNNGX IP Protection Framework

![commit badge](https://img.shields.io/badge/private-orange)
![commit badge](https://img.shields.io/badge/Spiking-red-Neural%20Network-blue)
![FGSM_GA_illustration_v2](./_img_src/SNNGX_cover.png)

## Encryption Simulation Description

**This code performs efficient SNNGX weight-bit (MSB) encryption and quantization on float32 Spiking neural networks and demonstrates biologically-plausible UNIVERSAL UTILITY on SNN models.** 

**We design an RRAM decryptor accelerator for secure neuromorphic SNN intellectual property (IP) and weight protections. Please refer to paper[link]**

![FGSM_GA_illustration_v2](./_img_src/SNNGX_result.png)

**This project was supported by AI Chip Center for Emerging Smart Systems (ACCESS) and the University of Hong Kong, EEE. Currently restricted to internal usage only.**

## Quick Start

*pytorch>=1.10.0+cu111*    
*tonic>=1.4.3*

```
git clone https://github.com/u3556440/SNNGX_qSNN_encryption.git
```


## Usage

*Dataset is not included in our repository. \
Please indicate your path to the Dataset (NMNIST/DVS128_Gesture) in the below bash commands.*


1. SNNGX Encryption (Genetic Algorithm):

```
cd Encryption_SNNGX
```
```
python3 __main__.py [-b] [--epsil] [--name] [--subset] [--mutate] [--gen] [--Dataset] [--Dpath]
```

Flag | VarType | Help
--- | --- | --- 
`--epsil` | int | Final number of bits to be converged by GA
`--by_layer` | bool | Boolean Flag (True for Layer-wise Encryption / False for All layers Encryption)
`--layer_idx` | int | Layer idx (starting from 0) for Layer-wise Encryption 
`-q`/`--qbit` | int | Quantized bit width of SNN (default: 8bit)
`--subset` | int | Total No. of Samples for encryption dataset
`-b`/`--batch` | int | Batch size for dataset, model, encryption dataset
`--mutate` | float | pixel-wise mutation probability (default: 0.05) 
`--gen` | int | No. of GA_generations
`--Dataset` | str | Available Dataset: NMNIST/DVS128_Gesture
`--Dpath` | str | Please input your local path to the dataset
`--seed` | int | Random Seed for Repeatable Experimental Result

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

Flag | VarType | Help
--- | --- | --- 
`-b`/`--batch` | int | Batch size for dataset and model
`-nb`/`--numbit` | int | No. of Random BITS flipped
`-nt`/`--numtrial` | int | No. of trial repetition
`--Dataset` | str | Available Dataset: "NMNIST"/"DVS128_Gesture"
`--Dpath` | str | Please input your local path to the dataset

You may wish to modify the follwing # commented code in __main__.py to change the search space 
from all layer to single target layer, and vice versa.

![searchSpace_random_flipping](https://github.com/u3556440/SNN_security/assets/56315946/bead64b1-8743-4b46-930f-82a63cfdfbd3)