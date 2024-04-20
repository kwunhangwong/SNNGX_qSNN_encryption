#!/bin/bash

# NMNIST (random Flipping 100 times) (Need to modify in __main__.py for flipping all layers/one layer)
python3 __main__.py -nb 400000 -nt 5 -q 8 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py -nb 450000 -nt 5 -q 8 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'

#layer wise (1)
#layer wise (2)

# DVS128_Gesture (random Flipping 100 times) (Need to modify in __main__.py for flipping all layers/one layer)
python3 __main__.py -nb 500000 -nt 5 -b 16 --Dataset "DVS128_Gesture" --Dpath '../../dataset/'
python3 __main__.py -nb 750000 -nt 5 -b 16 --Dataset "DVS128_Gesture" --Dpath '../../dataset/'
python3 __main__.py -nb 1000000 -nt 5 -b 16 --Dataset "DVS128_Gesture" --Dpath '../../dataset/'

