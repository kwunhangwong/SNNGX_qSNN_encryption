import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader  

from _Loading_All_Model import *
from Genetic_Algorithm_BITver import GA_BIT_flip_Untargeted, UNTARGETED_loader

import datetime
import argparse
import csv

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

transform = transforms.Compose([
    transforms.ToTensor(),  # convert PIL image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))
    ])  # normalize the input images

# load the test data
testset = torchvision.datasets.MNIST(root='../dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

annn = ANN_model()
checkpoint = torch.load('../pretrained_models/pretrained_binarised-mnist_ann_300e.t7',map_location=torch.device('cpu'))
annn.load_state_dict(checkpoint['net'])

UNTARGETED_loader = UNTARGETED_loader(test_set=testset,batch_size=batch_size,is_ANN=True)

#print(f"Before Untargeted Attack: {check_accuracy(test_loader,snnn)*100}% Accuracy")
start = datetime.datetime.now()
print(datetime.datetime.now())

with torch.no_grad():   #no need to cal grad
    Untargeted_attack = GA_BIT_flip_Untargeted(annn, UNTARGETED_loader, 
                                               epsil=epsil, mutate_chance=mutate_chance, n_generations=n_generations,
                                               BITS_by_layer=True, layer_type=nn.Linear, layer_idx=3)
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
