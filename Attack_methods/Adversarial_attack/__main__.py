import torch
import numpy as np
import random
import datetime
import argparse

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
parser.add_argument('-q','--qbit',type=int,metavar='target_quantized_bits',default=8,help='quantize weight from float32 to q bits') 

###############
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For model, test_loader and GA_attack_loader only') 
parser.add_argument('--Dataset',type=str,metavar='Target dataset',default="NMNIST", 
                    help= 'Please input: 1. "NMNIST" 2. DVS_Gesture only, their corresponding models will be selected automatically') 
parser.add_argument('--Dpath',type=str,metavar='path to dataset',default='../../BSNN_Project/N-MNIST_TRAINING/dataset', help='For dataset and model') 
parser.add_argument('--seed',type=int,metavar='random seed',default=10, help='np, torch, random...') 

###############
args = parser.parse_args()
setup_seed(args.seed)

quantized_bit = args.qbit

batch_size = args.batch
target_dataset = args.Dataset
dataset_path = args.Dpath

###############
# Constant
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device is {device}!!")

###############
# Adversarial Attack

def adver_val(model, test_loader, device, atk=None):
    correct = 0
    total = 0
    model.eval()
    for _, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        if atk is not None:
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    return final_acc


# MODEL 1
if (target_dataset == "NMNIST"):

    print("Loading Weights: ")
    weight_path = '../pretrained_weights_float32/pre_trained_normal-nmnist_snn_300e.t7'
    model = NMNIST_model(T_BIN=15).to(device)
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
    model = DVS128_model(T_BIN=15).to(device)
    checkpoint = torch.load(weight_path,map_location=device)
    model.load_state_dict(checkpoint['net'])
    quantize_weights_nbits(model,quantized_bit)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total trainable parameters: {num_params/2**20:.3f}M")
    print(f"This is a {quantized_bit}-bit quantized model: {num_params*quantized_bit/2**20:.3f}M bits")

    # Dataset (TEST and Subset Loader)
    _ , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)
    UNTARGETED_loader = UNTARGETED_loader(target=target_dataset,num_images=num_images,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)

else:
    raise ValueError("GA main: Target dataset not recognized. (NMNIST/DVS128_Gesture)")


print(f"Before Encryption: {check_accuracy(test_loader,model)*100}% Accuracy")
start = datetime.datetime.now()
print(datetime.datetime.now())



