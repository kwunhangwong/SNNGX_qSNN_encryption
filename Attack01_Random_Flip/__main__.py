import torch
import torch.nn as nn

import datetime
import argparse
import csv
from Random_Flipping_BITver import Random_flipping_all_Layers, Random_flipping_single_layer

import sys
sys.path.append('../binarization_models')

from _Loading_All_Dataset import *
from _Loading_All_Model import * 

###############
parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('-nb','--numbit',type=int,metavar='number_of_changed_BITS',required=True,help='# Random BITS flipped') 
parser.add_argument('-nt','--numtrial',type=int,metavar='number_of_random_trials',default=100,help='# Random Processes') 

###############
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For dataset and model') 
parser.add_argument('--Dataset',type=str,metavar='Target dataset',default="NMNIST", 
                    help= 'Please input: 1. "NMNIST" 2. "MNIST" only, their corresponding models will be selected automatically') 
parser.add_argument('--Dpath',type=str,metavar='path to dataset',default='../../BSNN_Project/N-MNIST_TRAINING/dataset', help='For dataset and model') 

###############
args = parser.parse_args()

num_bit = args.numbit
random_trial = args.numtrial

batch_size = args.batch
target_dataset = args.Dataset
dataset_path = args.Dpath

###############
# Constant
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device is {device}!!")

# MODEL 1
if (target_dataset == "NMNIST"):

    weight_path = '../pretrained_binary_weights/pretrained_binarised-nmnist_snn_300e_a92.t7'
    model = NMNIST_model(batch_size=batch_size).to(device)
    checkpoint = torch.load(weight_path,map_location=device)
    model.load_state_dict(checkpoint['net'])

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)

elif (target_dataset == "MNIST"):

    weight_path = '../pretrained_binary_weights/pretrained_binarised-mnist_ann_300e.t7'
    model = MNIST_model().to(device)
    checkpoint = torch.load(weight_path,map_location=device)
    model.load_state_dict(checkpoint['net'])

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)

else:
    raise ValueError("Random Flipping main: Target dataset not recognized. (NMNIST/MNIST)")


# Start testing with bits flipping
print(f"Before random flipping {num_bit} BITs: {check_accuracy(test_loader,model)*100}% Accuracy")
start = datetime.datetime.now()
print(start)

acc_list = []
for i in range(random_trial):
    print(f"Flipping {i+1}th times: ")

    # All Layer
    # model = Random_flipping_all_Layers(num_bit,model).to(device)

    # Single Layer
    model = Random_flipping_single_layer(num_bit,model,nn.Linear,3).to(device)

    acc_list+=[check_accuracy(test_loader,model).item()]

    # Restore the original weights 
    model.load_state_dict(checkpoint['net'])

end = datetime.datetime.now()
print(end)
print(f'Time: {end-start}')

# Save file 
name = f'flipping_{num_bit}_bit_Alllayer'

with open(name + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(acc_list)
