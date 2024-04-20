#!/bin/sh

# NMNIST 
# Testset size = 10000 # subset = 128/10000, 8/10000
# python3 __main__.py --epsil 50 --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 8 \
#     --Dataset 'NMNIST' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 1

# python3 __main__.py --epsil 50 --subset 128 --mutate 0.05 --qbit 8 --gen 120 -b 128 \
#     --Dataset 'NMNIST' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 1

# DVS_Gesture 
# Testset size = 264 # subset = 16/264
# python3 __main__.py --epsil 30 --subset 16 --mutate 0.05 --qbit 8 --gen 160 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 0

# python3 __main__.py --epsil 12 --subset 16 --mutate 0.05 --qbit 8 --gen 125 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 0

# python3 __main__.py --epsil 737 --subset 16 --mutate 0.05 --qbit 8 --gen 125 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 1

# python3 __main__.py --epsil 1475 --subset 16 --mutate 0.05 --qbit 8 --gen 125 -b 16 --seed 70  \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 2

# python3 __main__.py --epsil 2949 --subset 16 --mutate 0.05 --qbit 8 --gen 125 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 3

# python3 __main__.py --epsil 112 --subset 16 --mutate 0.05 --qbit 8 --gen 125 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 5

# python3 __main__.py --epsil 41943 --subset 16 --mutate 0.05 --qbit 8 --gen 125 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 4

# tmux capture-pane -pS -5000 > 1%.txt 

python3 __main__.py --epsil 35 --subset 16 --mutate 0.05 --qbit 8 --gen 95 -b 16 --seed 70 \
    --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 0

python3 __main__.py --epsil 2220 --subset 16 --mutate 0.05 --qbit 8 --gen 95 -b 16 --seed 70 \
    --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 1

python3 __main__.py --epsil 4425 --subset 16 --mutate 0.05 --qbit 8 --gen 95 -b 16 --seed 70  \
    --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 2

python3 __main__.py --epsil 8900 --subset 16 --mutate 0.05 --qbit 8 --gen 95 -b 16 --seed 70 \
    --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 3

python3 __main__.py --epsil 340 --subset 16 --mutate 0.05 --qbit 8 --gen 95 -b 16 --seed 70 \
    --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 5

python3 __main__.py --epsil 126000 --subset 16 --mutate 0.05 --qbit 8 --gen 95 -b 16 --seed 70 \
    --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 4

tmux capture-pane -pS -5000 > 3%.txt 

# 5% (seed 10, 42, 70)

# python3 __main__.py --epsil 50 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 0

# python3 __main__.py --epsil 3685 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 1

# python3 __main__.py --epsil 7375 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 --seed 70  \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 2

# python3 __main__.py --epsil 14745 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 3

# python3 __main__.py --epsil 560 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 5

# python3 __main__.py --epsil 209715 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 --seed 70 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 4

# tmux capture-pane -pS -5000 > 5%.txt 

# 8%
# python3 __main__.py --epsil 90 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 0

# python3 __main__.py --epsil 5900 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 1

# python3 __main__.py --epsil 11800 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 2

# python3 __main__.py --epsil 23600 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 3

# python3 __main__.py --epsil 900 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 5

# python3 __main__.py --epsil 340000 --subset 16 --mutate 0.05 --qbit 8 --gen 80 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 4

# tmux capture-pane -pS -5000 > 8%.txt

# 10%

# python3 __main__.py --epsil 29500 --subset 16 --mutate 0.05 --qbit 8 --gen 65 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 3

# python3 __main__.py --epsil 1200 --subset 16 --mutate 0.05 --qbit 8 --gen 65 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 5

# python3 __main__.py --epsil 450000 --subset 16 --mutate 0.05 --qbit 8 --gen 65 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 4

# python3 __main__.py --epsil 115 --subset 16 --mutate 0.05 --qbit 8 --gen 65 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 0

# python3 __main__.py --epsil 7370 --subset 16 --mutate 0.05 --qbit 8 --gen 65 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 1

# python3 __main__.py --epsil 14750 --subset 16 --mutate 0.05 --qbit 8 --gen 65 -b 16 \
#     --Dataset 'DVS128_Gesture' --Dpath '/home/edwin/code/dataset/' --by_layer True --layer_idx 2

# tmux capture-pane -pS -5000 > 10%.txt