#!/bin/sh

# NMNIST 
# Testset size = 10000 # subset = 128/10000, 8/10000
python3 __main__.py --epsil 50 --name "example_fitness_function" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 8 --Dataset "NMNIST" --Dpath '../../dataset/'
python3 __main__.py --epsil 50 --name "example_fitness_function" --subset 8 --mutate 0.05 --qbit 8 --gen 160 -b 8 --Dataset "NMNIST" --Dpath '../../dataset/'

# DVS_Gesture 
# Testset size = 264 # subset = 16/264
# python3 __main__.py --epsil 20 --name "ep_10_1Conv_DVSGesture" --subset 16 --mutate 0.05 --qbit 8 --gen 160 -b 16 --Dataset "NMNIST" --Dpath '../../dataset/'
# tmux capture-pane -pS -5000 > Signbitonly_bit8_activity.txt 
