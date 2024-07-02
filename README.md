# SNNGX IP Protection Framework

![commit badge](https://img.shields.io/badge/private-orange)
![commit badge](https://img.shields.io/badge/Spiking-Neural%20Network-red)
![SNNGX_illustration](./_img_src/SNNGX_cover.png)
**We design an RRAM decryptor accelerator for secure neuromorphic SNN intellectual property (IP) and weight protections. Please refer to paper[link]**

## Encryption Simulation Description
**This code performs efficient SNNGX genetic weight-bit (MSB) encryption for low overhead on neuromorphic accelerators. SNNGX method is universally applicable on differently pretrained SNN models and overcome gradient insensitivity problems.** 


![Protection_Performance](./_img_src/SNNGX_result.png)

**This research was supported by ACCESS - AI Chip Center for Emerging Smart Systems, sponsored by InnoHK funding, Hong Kong SAR**

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
python3 __main__.py [--epsil] [--by_layer] [--layer_idx] [--qbit] [--subset] [-b] [--mutate]  [--gen] [--Dataset] [--Dpath] [--seed]
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
`--Dataset` | str | Available Dataset: "NMNIST"/"DVS128_Gesture"
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
python3 __main__.py [-nb] [-nt] [--by_layer] [--layer_idx] [--qbit] [-b] [--Dataset] [--Dpath] [--seed]
```

Flag | VarType | Help
--- | --- | --- 
`-nb`/`--numbit` | int | Random BITS flipped %
`-nt`/`--numtrial` | int | No. of trial repetition 
`--by_layer` | bool | Boolean Flag (True for Layer-wise Encryption / False for All layers Encryption)
`--layer_idx` | int | Layer idx (starting from 0) for Layer-wise Encryption 
`-q`/`--qbit` | int | Quantized bit width of SNN (default: 8bit)
`-b`/`--batch` | int | Batch size for dataset and model
`--Dataset` | str | Available Dataset: "NMNIST"/"DVS128_Gesture"
`--Dpath` | str | Please input your local path to the dataset
`--seed` | int | Random Seed for Repeatable Experimental Result

You may wish to modify the follwing # commented code in __main__.py to change the search space 
from all layer to single target layer, and vice versa.

![searchSpace_random_flipping](https://github.com/u3556440/SNN_security/assets/56315946/bead64b1-8743-4b46-930f-82a63cfdfbd3)