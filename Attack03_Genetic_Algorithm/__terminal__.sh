#!/bin/sh

# NMNIST (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2, Layer type: nn.Linear)
python3 __main__.py --epsil 1 --name "ep_1lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 2 --name "ep_2lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 3 --name "ep_3lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 4 --name "ep_4lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 5 --name "ep_5lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 6 --name "ep_6lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 7 --name "ep_7lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 8 --name "ep_8lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 9 --name "ep_9lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 10 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'


python3 __main__.py --epsil 10 --name "ep_10lay2_NMNIST" --subset 128 --mutate 0.05 --qbit 8 --gen 160 -b 64 --Dataset "NMNIST" --Dpath '../../Torch_condaENV/Working_folder/dataset/'



# MNIST (Need to modify All layer attack or by layer attack in the main algorithm) (Default: Layer 2 (Wrong), Layer type: nn.Linear)
python3 __main__.py --epsil 1 --name "ep_1lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 2 --name "ep_2lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 3 --name "ep_3lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 4 --name "ep_4lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 5 --name "ep_5lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 6 --name "ep_6lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 7 --name "ep_7lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 8 --name "ep_8lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 9 --name "ep_9lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
python3 __main__.py --epsil 10 --name "ep_10lay2_MNIST" --subset 128 --mutate 0.05 --gen 160 -b 64 --Dataset "MNIST" --Dpath '../../BSNN_Project/N-MNIST_TRAINING/dataset'
tmux capture-pane -pS -5000 > MNIST_activity.txt 
