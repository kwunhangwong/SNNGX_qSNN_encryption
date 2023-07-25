# SNN_security

![Project Title](https://github.com/u3556440/SNN_security/assets/56315946/3c2a2b9f-9b6c-47cb-9f93-bdf6cd5ce16b)

![commit badge](https://img.shields.io/badge/private-8A2BE2)
![commit badge](https://img.shields.io/badge/Binary-Neural%20Network-blue)


## Description

**This project creates BIT-flip attack against binarised weight of deep neural network and other machine learning models with flipping minimum vulnerable BITs.** 

There are 3 attack methods in this repository and some pretrained binary weight models over NMNIST and MNIST. 

Apart from random flipping tests, we proposed two more optimzation methods to attck the most vulnerable bits in the binarised ANN/SNN models: FGSM-based attack (gradient) and Genetic Algorithm-based attack (non-gradient) respectively. 

![attack methods]((https://github.com/u3556440/SNN_security/assets/56315946/048b3b29-6fbb-426c-9c17-3ef3aa7bf551))


This project was supported by the University of Hong Kong, EEE. \
Currently restricted to internal usage only.

## Usage

### Dataset is not included in our repository. Please indicate your path to the Dataset (NMNIST/MNIST) in the below bash commands.

1. Random Flipping Attack:

```
Usage: command [options] [arguments]
```

Flag | Metavar | Default |Help
--- | --- | --- | ---
`-b`/`--batch` | batch_size | **Batch size for dataset and model**
`-nb`/`--numbit` | number_of_flipped_BITS | **# of Random BITS flipped**
`-nt`/`--numtrial` | number_of_random_trials | **# of trial repetition**


2. FGSM-based weight Attack:

```
Usage: command [options] [arguments]
```

3. Genetic Algorithm weight Attack:

```
Usage: command [options] [arguments]
```




