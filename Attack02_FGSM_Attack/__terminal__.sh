#!/bin/sh

# NMNIST (Search_by_layer)
python3 __main__.py --expected_bit 10 --topk 8 -b 128 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'

# MNIST (Search_by_layer)
python3 __main__.py --expected_bit 10 --topk 8 -b 128 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'