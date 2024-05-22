import torch
import torch.nn as nn
import numpy as np
import random

import datetime
import argparse
import csv
from Random_Flipping_BITver import Random_flipping_single_layer, Random_flipping_all_Layers

import numpy as np
import sys
sys.path.append('../quantization_utils')

from _Loading_All_Model import *
from _Loading_All_Dataloader import * 
from quantization import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

###############
parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('-nb','--numbit',type=float,metavar='%_of_changed_BITS',required=True,help='% of Random BITS flipped in target layer') 
parser.add_argument('-nt','--numtrial',type=int,metavar='number_of_random_trials',default=100,help='# Random Processes') 

###############
parser.add_argument('--by_layer',action="store_true", help = 'Encryption on only one layer or whole model') 
parser.add_argument('--layer_idx',type=int,metavar='layer index',default=0,help = 'Encryption target layer if only one layer (ranked by nn.Linear and nn.Conv2d)') 
parser.add_argument('-q','--qbit',type=int,metavar='target_quantized_bits',default=8,help='quantize weight from float32 to q bits') 

###############
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For dataset and model') 
parser.add_argument('--Dataset',type=str,metavar='Target dataset',default="NMNIST", 
                    help= 'Please input: 1. "NMNIST" 2. "DVS128_Gesture", their corresponding model and loader will be selected automatically') 
parser.add_argument('--Dpath',type=str,metavar='path to dataset',default='Please INPUT you path to dataset', help='For dataset and model') 
parser.add_argument('--seed',type=int,metavar='random seed',default=70, help='np, torch, random...') 

###############
args = parser.parse_args()
setup_seed(args.seed)

num_bit = args.numbit
random_trial = args.numtrial

by_layer = args.by_layer
layer_idx = args.layer_idx
quantized_bit = args.qbit

batch_size = args.batch
target_dataset = args.Dataset
dataset_path = args.Dpath

###############
# Constant
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device is {device}!!")

# MODEL 1
if (target_dataset == "NMNIST"):

    weight_path = '../pretrained_weights_float32/pre_trained_normal-nmnist_snn_300e.t7'
    model = NMNIST_model().to(device)

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)

elif (target_dataset == "DVS128_Gesture"):

    weight_path = '../pretrained_weights_float32/pretrained_DVS_csnn_128e_91a.t7'
    model = DVS128_model().to(device)

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)

else:
    raise ValueError("Random Flipping main: Target dataset not recognized. (NMNIST/DVS128_Gesture)")


# quantized the model first after loading 
checkpoint = torch.load(weight_path,map_location=device)
model.load_state_dict(checkpoint['net'])
quantize_weights_nbits(model,quantized_bit)

# Start testing with bits flipping
print(f"Before Random Encryption, {quantized_bit}-bit model Acc: {check_accuracy(test_loader,model)*100}% Accuracy")
start = datetime.datetime.now()
print(start)

acc_list = []
for i in range(random_trial):
    print(f"Flipping {num_bit*100}(%) bits {i+1}th times: ")

    if by_layer:
       # Single Layer (all weight bits)
        model = Random_flipping_single_layer(num_bit,model,quantized_bit,layer_idx).to(device)
    
    else:
        # All Layer (all weight bits)
        model = Random_flipping_all_Layers(num_bit,model,quantized_bit).to(device)
 
    acc_list+=[check_accuracy(test_loader,model).item()]

    # Restore the original weights 
    model.load_state_dict(checkpoint['net'])
    quantize_weights_nbits(model,quantized_bit)


end = datetime.datetime.now()
print(end)
print(f'Time: {end-start}')

# Save file 
name = f'flipping_{num_bit}_Bits'

with open(name + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(acc_list)
