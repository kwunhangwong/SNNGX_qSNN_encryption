#!/bin/sh


# NMNIST (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2, Layer type: nn.Linear)
python3 __main__.py --epsil 1 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 2 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 3 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 4 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 5 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 6 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 7 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 8 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 9 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 10 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
tmux capture-pane -pS -5000 > NMNIST_activity.txt 

# MNIST (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2 (Wrong), Layer type: nn.Linear)
python3 __main__.py --epsil 1 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 2 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 3 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 4 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 5 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 6 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 7 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 8 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 9 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 10 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
tmux capture-pane -pS -5000 > MNIST_activity.txt 
