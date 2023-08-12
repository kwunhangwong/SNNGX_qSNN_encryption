#!/bin/bash

# NMNIST (random Flipping 100 times) (Need to modify in __main__.py for flipping all layers/one layer)
python3 __main__.py -nb 10 -nt 100 -q 8 -b 64 --Dataset "NMNIST" --Dpath "../../Torch_condaENV/Working_folder/dataset/"
python3 __main__.py -nb 20 -nt 100 -q 8 -b 64 --Dataset "NMNIST" --Dpath "../../Torch_condaENV/Working_folder/dataset/"
python3 __main__.py -nb 30 -nt 100 -q 8 -b 64 --Dataset "NMNIST" --Dpath "../../Torch_condaENV/Working_folder/dataset/"
python3 __main__.py -nb 40 -nt 100 -q 8 -b 64 --Dataset "NMNIST" --Dpath "../../Torch_condaENV/Working_folder/dataset/"


# DVS128_Gesture (random Flipping 100 times) (Need to modify in __main__.py for flipping all layers/one layer)





# MNIST (random Flipping 100 times) (Need to modify in __main__.py for flipping all layers/one layer)
python3 __main__.py -nb 10 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 20 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 30 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 40 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 50 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 60 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 70 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 80 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 90 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py -nb 100 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
