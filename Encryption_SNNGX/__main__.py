import torch
import numpy as np
import random

import datetime
import argparse
from GA_BITver_minBIT_layer import SNNGX_BIT_Encryption

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

###############
parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('--epsil',type=int,metavar='GA_epsilon',default=10,help = 'The final number of bits to be converge by GA') 
parser.add_argument('--by_layer',type=bool,metavar='encrypt by layer',default=True,help = 'Encryption on only one layer or whole model') 
parser.add_argument('--layer_idx',type=int,metavar='layer index',default=0,help = 'Encryption target layer if only one layer (ranked by nn.Linear and nn.Conv2d)') 
parser.add_argument('-q','--qbit',type=int,metavar='target_quantized_bits',default=8,help='quantize weight from float32 to q bits') 

###############
parser.add_argument('--subset',type=int,metavar='Number of subset images',default=128, help = 'No. of Samples for calculating fitness function') 
parser.add_argument('--mutate',type=float,metavar='GA_mutate_chance',default=0.005) 
parser.add_argument('--gen',type=int,metavar='# GA_generations',default=160) 

###############
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For model, test_loader and GA_attack_loader only') 
parser.add_argument('--Dataset',type=str,metavar='Target dataset',default="NMNIST", 
                    help= 'Please input: 1. "NMNIST" 2. DVS_Gesture only, their corresponding models will be selected automatically') 
parser.add_argument('--Dpath',type=str,metavar='path to dataset',default='../../BSNN_Project/N-MNIST_TRAINING/dataset', help='For dataset and model') 
parser.add_argument('--seed',type=int,metavar='random seed',default=10, help='np, torch, random...') 

###############
args = parser.parse_args()
setup_seed(args.seed)

epsil = args.epsil
by_layer = args.by_layer
layer_idx = args.layer_idx
quantized_bit = args.qbit

num_images = args.subset
mutate_chance = args.mutate
n_generations = args.gen

batch_size = args.batch
target_dataset = args.Dataset
dataset_path = args.Dpath

###############
# Constant
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device is {device}!!")

# MODEL 1
if (target_dataset == "NMNIST"):

    print("Loading Weights: ")
    weight_path = '../pretrained_weights_float32/pre_trained_normal-nmnist_snn_300e.t7'
    model = NMNIST_model().to(device)
    checkpoint = torch.load(weight_path,map_location=device)
    model.load_state_dict(checkpoint['net'])
    quantize_weights_nbits(model,quantized_bit)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total trainable parameters: {num_params/2**20:.3f}M")
    print(f"This is a {quantized_bit}-bit quantized model: {num_params*quantized_bit/2**20:.3f}M bits")

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)
    UNTARGETED_loader = UNTARGETED_loader(target=target_dataset,num_images=num_images,batch_size=batch_size,T_BIN=15,dataset_path =dataset_path)

elif (target_dataset == "DVS128_Gesture"):

    weight_path = '../pretrained_weights_float32/pretrained_DVS_csnn_128e_91a.t7'
    model = DVS128_model().to(device)
    checkpoint = torch.load(weight_path,map_location=device)
    model.load_state_dict(checkpoint['net'])
    quantize_weights_nbits(model,quantized_bit)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total trainable parameters: {num_params/2**20:.3f}M")
    print(f"This is a {quantized_bit}-bit quantized model: {num_params*quantized_bit/2**20:.3f}M bits")

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)
    UNTARGETED_loader = UNTARGETED_loader(target=target_dataset,num_images=num_images,batch_size=batch_size,T_BIN=15,dataset_path =dataset_path)

else:
    raise ValueError("GA main: Target dataset not recognized. (NMNIST/DVS128_Gesture)")


print(f"Before Untargeted Attack: {check_accuracy(test_loader,model)*100}% Accuracy")
start = datetime.datetime.now()
print(datetime.datetime.now())

with torch.no_grad():   #no need to cal grad
    Untargeted_attack = SNNGX_BIT_Encryption(model, UNTARGETED_loader, 
                                             epsil=epsil, 
                                             mutate_chance=mutate_chance, 
                                             n_generations=n_generations,
                                             BITS_by_layer=by_layer, 
                                             layer_idx=layer_idx,
                                             qbits=quantized_bit)
    adv_model, advBIT, numBIT ,fitness = Untargeted_attack.main()
    
end = datetime.datetime.now()
print(datetime.datetime.now())
final_result = check_accuracy(test_loader,adv_model)
print(f"After Untargeted Attack: {final_result*100:.2f}% Accuracy, BITS Flipped:{advBIT} out of {numBIT}")
print(f"Time = {end-start}")
print(fitness)

# Save adversarial model
print("Encrypted model saving......")
names = target_dataset
state = {
    'net': model.state_dict(),
    'advBIT': advBIT,
    'numBIT': numBIT,
    'fitness_score': fitness,
}

torch.save(state, './encrypted_' + names +'.t7')
