#!/bin/bash

# NMNIST (random Flipping 100 times) (Need to modify in __main__.py for flipping all layers/one layer)
python3 __main__.py -nb 400000 -nt 5 -q 8 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py -nb 450000 -nt 5 -q 8 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'



#+=_______not processed________=+#
#layer wise (1)
# python3 __main__.py -nb 50 -nt 100 -q 8 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
#layer wise (2)

# DVS128_Gesture (random Flipping 100 times) (Need to modify in __main__.py for flipping all layers/one layer)

# python3 __main__.py -nb 272 -nt 50 -b 32 --Dataset "DVS128_Gesture" --Dpath '../../dataset/'



# MNIST (random Flipping 100 times) (Need to modify in __main__.py for flipping all layers/one layer)
# python3 __main__.py -nb 10 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 20 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 30 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 40 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 50 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 60 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 70 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 80 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 90 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
# python3 __main__.py -nb 100 -nt 100 -b 64 --Dataset "DVS128_Gesture" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
