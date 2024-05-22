#!/bin/sh

# NMNIST 
# Output Layer: layer_idx=1
python3 __main__.py --epsil 50 --subset 16 --mutate 0.05 --qbit 8 --gen 125 -b 16 --seed 70\
    --Dataset 'NMNIST' --Dpath '/home/edwin/code/dataset/' --by_layer --layer_idx 1

python3 __main__.py --epsil 50 --subset 16 --mutate 0.05 --qbit 8 --gen 125 -b 16 --seed 70\
    --Dataset 'NMNIST' --Dpath '/home/edwin/code/dataset/' --by_layer --layer_idx 1

# DVS_Gesture 
# Input Layer: Layer_idx=0
python3 __main__.py --epsil 30 --subset 16 --mutate 0.05 --qbit 8 --gen 95 -b 16 --seed 70\
    --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer --layer_idx 0

python3 __main__.py --epsil 30 --subset 16 --mutate 0.05 --qbit 8 --gen 95 -b 16 --seed 70 \
    --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer --layer_idx 0