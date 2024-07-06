#!/bin/bash

# NMNIST (random Flipping 100 times) 
python3 __main__.py -nb 0.086 -nt 100 -q 8 -b 64 --Dataset "NMNIST" \
    --Dpath '../../dataset/' 

# python3 __main__.py -nb 0.086 -nt 100 -q 8 -b 64 --Dataset "NMNIST" \
    # --Dpath '../../dataset/' --by_layer --layer_idx 1
    
# DVS128_Gesture (random Flipping 100 times) 
# python3 __main__.py -nb 0.045 -nt 100 -q 8 -b 16 --Dataset "DVS128_Gesture"\
     # --Dpath '../../dataset/' 

# python3 __main__.py -nb 0.045 -nt 100 -q 8 -b 16 --Dataset "DVS128_Gesture"\
     # --Dpath '../../dataset/' --by_layer --layer_idx 0

