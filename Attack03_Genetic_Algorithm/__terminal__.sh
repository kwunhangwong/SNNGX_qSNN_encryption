#!/bin/sh

# NMNIST (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2, Layer type: nn.Linear)

python3 __main__.py --epsil 10 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 20 --name "ep_20lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 30 --name "ep_30lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 40 --name "ep_40lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 50 --name "ep_50lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 60 --name "ep_60lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 60 --name "ep_60lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../dataset/'
tmux capture-pane -pS -5000 > Signbitonly_bit8_activity.txt 


# DVS_Gesture (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2 (Wrong), Layer type: nn.Linear)

