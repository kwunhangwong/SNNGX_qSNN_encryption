import torch
import torch.nn as nn
from torch.utils.data import DataLoader  

import datetime
import argparse
import csv
from Genetic_Algorithm_BITver import GA_BIT_flip_Untargeted, UNTARGETED_loader

import sys
sys.path.append('../binarization_models')

from _Loading_All_Dataset import *
from _Loading_All_Model import * 

###############
parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('--epsil',type=int,metavar='GA_epsilon',default=10,help = 'The final number of bits to be converge by GA') 
parser.add_argument('--name',type=str,metavar='Documenting_fitness',default="unnamed") 

###############
parser.add_argument('--subset',type=int,metavar='Number of subset images',default=128, help = 'No. of Samples for calculating fitness function') 
parser.add_argument('--mutate',type=float,metavar='GA_mutate_chance',default=0.005) 
parser.add_argument('--gen',type=int,metavar='# GA_generations',default=160) 

###############
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For dataset, model, GA-subset') 
parser.add_argument('--Dataset',type=str,metavar='Target dataset',default="NMNIST", 
                    help= 'Please input: 1. "NMNIST" 2. "MNIST" only, their corresponding models will be selected automatically') 
parser.add_argument('--Dpath',type=str,metavar='path to dataset',default='../../BSNN_Project/N-MNIST_TRAINING/dataset', help='For dataset and model') 

###############
args = parser.parse_args()

epsil = args.epsil
name = args.name

num_images = args.subset
mutate_chance = args.mutate
n_generations = args.gen

batch_size = args.batch
target_dataset = args.Dataset
dataset_path = args.Dpath

###############
# Constant
device = torch.device('cuda')
print(f"Current device is {device}!!")

# MODEL 1

if (target_dataset == "NMNIST"):

    weight_path = '../pretrained_binary_weights/pretrained_binarised-nmnist_snn_300e_a92.t7'
    model = NMNIST_model(batch_size=batch_size).to(device)
    checkpoint = torch.load(weight_path,map_location=device)
    model.load_state_dict(checkpoint['net'])

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)
    UNTARGETED_loader = UNTARGETED_loader(target=target_dataset,num_images=num_images,batch_size=batch_size,T_BIN=15,dataset_path =dataset_path)

elif (target_dataset == "MNIST"):

    weight_path = '../pretrained_binary_weights/pretrained_binarised-mnist_ann_300e.t7'
    model = MNIST_model().to(device)
    checkpoint = torch.load(weight_path,map_location=device)
    model.load_state_dict(checkpoint['net'])

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)
    UNTARGETED_loader = UNTARGETED_loader(target=target_dataset,num_images=num_images,batch_size=batch_size,T_BIN=15,dataset_path =dataset_path)

else:
    raise ValueError("GA main: Target dataset not recognized. (NMNIST/MNIST)")


print(f"Before Untargeted Attack: {check_accuracy(test_loader,model)*100}% Accuracy")
start = datetime.datetime.now()
print(datetime.datetime.now())

with torch.no_grad():   #no need to cal grad
    Untargeted_attack = GA_BIT_flip_Untargeted(model, UNTARGETED_loader, 
                                               epsil=epsil, mutate_chance=mutate_chance, n_generations=n_generations,
                                               BITS_by_layer=True, layer_type=nn.Linear, layer_idx=2)
    adv_model, advBIT, numBIT ,fitness = Untargeted_attack.main()
    
end = datetime.datetime.now()
print(datetime.datetime.now())
print(f"After Untargeted Attack: {check_accuracy(test_loader,adv_model)*100:.2f}% Accuracy, BITS Flipped:{advBIT} out of {numBIT}")
print(f"Time = {end-start}")

print(fitness)

# Save file 
with open(name + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fitness)
