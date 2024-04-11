import torch
import torch.nn as nn

import datetime
import argparse
import csv
from Random_Flipping_BITver import Random_flipping_all_Layers, Random_flipping_single_layer

import sys
sys.path.append('../../quantization_utils')

from _Loading_All_Model import *
from _Loading_All_Dataloader import * 
from quantization import *

###############
parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('-q','--qbit',type=int,metavar='target_quantized_bits',default=8,help='quantize weight from float32 to q bits') 

###############
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For dataset and model') 
parser.add_argument('--Dataset',type=str,metavar='Target dataset',default="NMNIST", 
                    help= 'Please input: 1. "NMNIST" 2. "DVS128_Gesture", their corresponding model and loader will be selected automatically') 
parser.add_argument('--Dpath',type=str,metavar='path to dataset',default='Please INPUT you path to dataset', help='For dataset and model') 

###############
args = parser.parse_args()

num_bit = args.numbit
random_trial = args.numtrial
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
    model = NMNIST_model(batch_size=batch_size).to(device)

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)

elif (target_dataset == "DVS128_Gesture"):

    weight_path = '../pretrained_weights_float32/pretrained_DVS_csnn_128e_91a.t7'
    model = DVS128_model(batch_size=batch_size).to(device)

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)

else:
    raise ValueError("Random Flipping main: Target dataset not recognized. (NMNIST/DVS128_Gesture)")


# quantized the model first after loading 
checkpoint = torch.load(weight_path,map_location=device)
model.load_state_dict(checkpoint['net'])
quantize_weights_nbits(model,quantized_bit)

#Model Training
max_acc=0
Test_acc=[]
for epoch in range(num_epochs):
    
    for batch_idx,(images, labels) in enumerate(train_loader):  #Each Batch # The enumerate() method adds a counter to an iterable

        model.zero_grad()
        optimizer.zero_grad()

        # Train on Cuda
        images = images.float().to(device)
        labels = labels.to(device)

        #print(images.shape) #=> 4x32x2x34x34 = longest_time_seq_in_the_batch x batch_size x channels x height x width 
        labels_onehot = F.one_hot(labels, 10).float()  # AABBBCC -> 100, 100, 010, 010, 010, 001, 001        
        
        # T x N x 2 x 34 x 34 => N x 10
        out_firing = model(images)

        # Loss function => MSE so one_hots for L2 distance error
        loss = criterion(out_firing, labels_onehot)
        loss.backward()
        optimizer.step() 
    
    # Check acc
    print(f'Epoch Done: {epoch}')
    curr_acc = check_accuracy(test_loader,model).item()
    Test_acc +=[curr_acc]

    if curr_acc >= max_acc:
        max_acc = curr_acc
        print("State-of-the-art saving......")
        names = 'binarised-nmnist_snn_300e'
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'acc_record': Test_acc,
        }
        torch.save(state, './pretrained_' + names +'.t7')

print("Finished Training!!")
