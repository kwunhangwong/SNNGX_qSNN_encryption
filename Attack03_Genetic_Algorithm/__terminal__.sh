#!/bin/sh

# NMNIST (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2, Layer type: nn.Linear)
# Testset size = 10000 # subset = 128, 64, 32, 16, 8
python3 __main__.py --epsil 40 --name "ep_50lay2_sub128NMNIST_2try" --subset 32 --mutate 0.05 --qbit 6 --gen 160 -b 32 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 40 --name "ep_10lay2_sub128NMNIST_2try" --subset 32 --mutate 0.05 --qbit 6 --gen 160 -b 32 --Dataset "NMNIST" --Dpath '../../dataset/'

python3 __main__.py --epsil 40 --name "ep_20lay2_sub128NMNIST_3" --subset 32 --mutate 0.05 --qbit 6 --gen 160 -b 32 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 40 --name "ep_20lay2_sub128NMNIST_4" --subset 32 --mutate 0.05 --qbit 6 --gen 160 -b 32 --Dataset "NMNIST" --Dpath '../../dataset/'

# python3 __main__.py --epsil 30 --name "ep_30lay2_sub128NMNIST_3" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 128 --Dataset "NMNIST" --Dpath '../../dataset/'
# python3 __main__.py --epsil 40 --name "ep_40lay2_sub128NMNIST_3" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 128 --Dataset "NMNIST" --Dpath '../../dataset/'
# python3 __main__.py --epsil 50 --name "ep_50lay2_sub128NMNIST_3" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 128 --Dataset "NMNIST" --Dpath '../../dataset/'
# tmux capture-pane -pS -5000 > Signbitonly_bit8_128activity.txt 

# DVS_Gesture (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2 (Wrong), Layer type: nn.Linear)
# Testset size = 264 # subset = 32/264
# python3 __main__.py --epsil 10 --name "ep_10_1Conv_DVSGesture" --subset 16 --mutate 0.05 --qbit 8 --gen 160 -b 8 --Dataset "NMNIST" --Dpath '../../dataset/'

# python3 __main__.py --epsil 10 --name "ep_10_1Conv_DVSGesture" --subset 32 -ub 16 --mutate 0.05 --qbit 8 --gen 160 -b 16 --Dataset "NMNIST" --Dpath '../../dataset/'
# python3 __main__.py --epsil 20 --name "ep_20_1Conv_DVSGesture" --subset 32 -ub 16 --mutate 0.05 --qbit 8 --gen 160 -b 16 --Dataset "NMNIST" --Dpath '../../dataset/'
# python3 __main__.py --epsil 30 --name "ep_30_1Conv_DVSGesture" --subset 32 -ub 16 --mutate 0.05 --qbit 8 --gen 160 -b 16 --Dataset "NMNIST" --Dpath '../../dataset/'
# python3 __main__.py --epsil 40 --name "ep_40_1Conv_DVSGesture" --subset 32 -ub 16 --mutate 0.05 --qbit 8 --gen 160 -b 16 --Dataset "NMNIST" --Dpath '../../dataset/'
# python3 __main__.py --epsil 50 --name "ep_50_1Conv_DVSGesture" --subset 32 -ub 16 --mutate 0.05 --qbit 8 --gen 160 -b 16 --Dataset "NMNIST" --Dpath '../../dataset/'
# tmux capture-pane -pS -5000 > Signbitonly_bit8_activity.txt 
