#!/bin/sh

# NMNIST (Search_by_layer)
python3 __main__.py --expected_bit 10 --topk 8 --qbit 4 -b 128 --Dataset "NMNIST" --Dpath '../../Torch_condaENV/Working_folder/dataset/'
python3 __main__.py --expected_bit 10 --topk 8 --qbit 8 -b 128 --Dataset "NMNIST" --Dpath '../../Torch_condaENV/Working_folder/dataset/'


# MNIST (Search_by_layer)
python3 __main__.py --expected_bit 10 --topk 8 -b 128 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'