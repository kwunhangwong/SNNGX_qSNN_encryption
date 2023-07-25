import torch

import datetime
import argparse
from Fast_Grad_Sign_BITver import FGSM_Untargeted_BIT_flip

import sys
sys.path.append('../binarization_models')

from _Loading_All_Dataset import *
from _Loading_All_Model import * 


###############
parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('--expected_bit',type=int,metavar='expected-number_of_changed_BITS',default=20,help='indeterministic number of flipped bits, must be more than topk') 
parser.add_argument('--topk',type=int,metavar='top k of bits with greatest gradient to be flipped',default=10,help='Flipping process with FGSM-based theory') 

###############
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For dataset, model and FGSM samples') 
parser.add_argument('--Dataset',type=str,metavar='Target dataset',default="NMNIST", 
                    help= 'Please input: 1. "NMNIST" 2. "MNIST" only, their corresponding models will be selected automatically') 
parser.add_argument('--Dpath',type=str,metavar='path to dataset',default='../../BSNN_Project/N-MNIST_TRAINING/dataset', help='For dataset and model') 

###############
args = parser.parse_args()

expected_bit = args.expected_bit
topk = args.topk

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
    train_loader , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)
    FGSM_attack = FGSM_Untargeted_BIT_flip(model, k_top=topk,expected_bits=expected_bit, iteration=10,is_ANN=False)

elif (target_dataset == "MNIST"):

    weight_path = '../pretrained_binary_weights/pretrained_binarised-mnist_ann_300e.t7'
    model = MNIST_model().to(device)
    checkpoint = torch.load(weight_path,map_location=device)
    model.load_state_dict(checkpoint['net'])

    # Dataset (TEST and Subset Loader)
    train_loader , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)
    FGSM_attack = FGSM_Untargeted_BIT_flip(model, k_top=topk,expected_bits=expected_bit, iteration=10,is_ANN=True)

else:
    raise ValueError("Random Flipping main: Target dataset not recognized. (NMNIST/MNIST)")

# Start testing before FGSM attack
print(f"Before Untargeted FGSM-Attack: {check_accuracy(test_loader,model)*100}% Accuracy")
start = datetime.datetime.now()
print(datetime.datetime.now())

# Using first batach of train_loader for FGSM attacks
for _, (data, target) in enumerate(train_loader):
    # Get First Batch of train_loader
    data = data.float().to(device)
    target = target.to(device)
    # Override the target to prevent label leaking
    _, target = model(data).max(1)
    break

# Attack finding minimum bitflip
bit_flipped, numlayer_weight = FGSM_attack.progressive_bit_search(data,target)

print(f'Total BITS flipped: {bit_flipped}')  
print(f"After Untargeted FGSM-Attack: {check_accuracy(test_loader,model)*100:.2f}% Accuracy, BITS Flipped:{bit_flipped} out of {numlayer_weight}")

end = datetime.datetime.now()
print(end)
print(f"Time = {end-start}")

