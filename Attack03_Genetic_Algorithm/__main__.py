import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader  

import datetime
import argparse
import csv
from Genetic_Algorithm_BITver import GA_BIT_flip_Untargeted, UNTARGETED_loader

import sys
sys.path.append('../binarization_models')

from _Loading_All_Model import *



parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('--name',type=str,metavar='Documenting_fitness',default="unnamed") 
parser.add_argument('--epsil',type=int,metavar='GA_epsilon',default=100) 
parser.add_argument('--mutate',type=float,metavar='GA_mutate_chance',default=0.005) 
parser.add_argument('--gen',type=int,metavar='# GA_generations ',default=10) 
args = parser.parse_args()

name = args.name
epsil = args.epsil
mutate_chance = args.mutate
n_generations = args.gen

# Constant
batch_size = 64

device = torch.device('cuda')
print(f"Current device is {device}!!")

# MODEL 1
snnn = SNN_model(batch_size=batch_size).to(device)
checkpoint = torch.load('../pretrained_models/pre_trained_binarised-nmnist_snn_300e_a92.t7',map_location=device)
snnn.load_state_dict(checkpoint['net'])

# Dataset (UNTARGETED and TEST Loader)
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size, n_time_bins=15)])
    
testset = tonic.datasets.NMNIST(save_to="../dataset", transform=frame_transform, train=False)
test_loader = DataLoader(
    dataset = testset,
    batch_size= batch_size,
    collate_fn= tonic.collation.PadTensors(batch_first=False),
    shuffle = True,
    drop_last=True
)

UNTARGETED_loader = UNTARGETED_loader(test_set=testset,batch_size=batch_size,is_ANN=False)

#print(f"Before Untargeted Attack: {check_accuracy(test_loader,snnn)*100}% Accuracy")
start = datetime.datetime.now()
print(datetime.datetime.now())

with torch.no_grad():   #no need to cal grad
    Untargeted_attack = GA_BIT_flip_Untargeted(snnn, UNTARGETED_loader, 
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
