#!/bin/sh

# NMNIST (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2, Layer type: nn.Linear)

python3 __main__.py --epsil 60 --name "ep_60lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 70 --name "ep_70lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 80 --name "ep_80lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 90 --name "ep_90lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 100 --name "ep_100lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 150 --name "ep_150lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
tmux capture-pane -pS -5000 > MNIST_activity.txt 


# MNIST (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2 (Wrong), Layer type: nn.Linear)

