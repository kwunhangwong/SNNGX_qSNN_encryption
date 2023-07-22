from Loading_model import SNN_model, ActivationFun, act_fun, mem_update, check_accuracy, VTH, DECAY, alpha
from Random_flipping import Random_flipping_all_Layers, Random_flipping_single_layer

import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader  
import datetime
import csv

import argparse

parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('-nb','--numbit',type=int,metavar='number_of_changed_BITS',required=True,help='# Random BITS flipped') 
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For dataset and model') 
parser.add_argument('-nt','--numtrial',type=int,metavar='number_of_random_trials',default=100,help='# Random Processes') 
args = parser.parse_args()

num_bit = args.numbit
batch_size = args.batch
random_trial = args.numtrial

# Dataset - NMNIST
# 2 x 34 x 34
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size, n_time_bins=15)])
    
trainset = tonic.datasets.NMNIST(save_to="../dataset",transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to="../dataset", transform=frame_transform, train=False)

#collation整理 => pad out填充 shorting recordings to have same dimension
train_loader = DataLoader(
    dataset = trainset,
    batch_size= batch_size,
    collate_fn= tonic.collation.PadTensors(batch_first=False),
    shuffle = True,
    drop_last=True
)

test_loader = DataLoader(
    dataset = testset,
    batch_size= batch_size,
    collate_fn= tonic.collation.PadTensors(batch_first=False),
    shuffle = True,
    drop_last=True
)

# MODEL 1: Binarised SNN (type fc)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"NOW device is using: {device}")

snnn = SNN_model(batch_size=batch_size).to(device)
checkpoint = torch.load('../pretrained_models/pre_trained_binarised-nmnist_snn_300e_a92.t7',map_location=torch.device('cuda'))
snnn.load_state_dict(checkpoint['net'])


# Start testing with bits flipping
start = datetime.datetime.now()
print(start)

# print("Before attack, check if your model correct: ",check_accuracy(test_loader,snnn))

acc_list = []
for i in range(random_trial):
    print("Flipping: ")

    # All Layer
    model = Random_flipping_all_Layers(num_bit,snnn).to(device)

    # Single Layer
    # model = Random_flipping_single_layer(num_bit,snnn,nn.Linear,2)

    acc_list+=[check_accuracy(test_loader,model).item()]

    snnn.load_state_dict(checkpoint['net'])

end = datetime.datetime.now()
print(end)
print(f'Time: {end-start}')

# Save file 
name = f'flipping_{num_bit}_bit_Alllayer'

with open(name + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(acc_list)
