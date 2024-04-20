import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import argparse
import random
import numpy as np
import json

import datetime
import sys
sys.path.append('../../quantization_utils')

from _Loading_All_Model import *
from _Loading_All_Dataloader import * 
from quantization import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def write_log(new_log, directory):
    log_file = open(directory, "a")

    # Write the output to the log file as JSON
    json.dump(new_log, log_file)
    log_file.write("\n")  # Add a newline between entries (optional)

    log_file.close()
    return None

###############
parser = argparse.ArgumentParser()

# Required input: num_bit
parser.add_argument('-q','--qbit',type=int,metavar='target_quantized_bits',default=8,help='quantize weight from float32 to q bits') 
parser.add_argument('--subset',type=int,metavar='Number of subset images',default=128, help = 'No. of Samples for calculating fitness function') 

###############
parser.add_argument('-b','--batch',type=int,metavar='batch_size',default=64,help='For dataset and model') 
parser.add_argument('--Dataset',type=str,metavar='Target dataset',default="NMNIST", 
                    help= 'Please input: 1. "NMNIST" 2. "DVS128_Gesture", their corresponding model and loader will be selected automatically') 
parser.add_argument('--Dpath',type=str,metavar='path to dataset',default='Please INPUT you path to dataset', help='For dataset') 
parser.add_argument('--encrypt_model',type=str,metavar='path to encrypted model',default='../../Encryption_SNNGX/encrypted_DVS128_Gesture.t7', help='model path should align with --Dataset') 


###############
args = parser.parse_args()
setup_seed(10)

quantized_bit = args.qbit
num_images = args.subset

batch_size = args.batch
target_dataset = args.Dataset
dataset_path = args.Dpath

###############
# Constant
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device is {device}!!")

# MODEL 1
if (target_dataset == "NMNIST"):

    encrypted_weight_path = '../pretrained_weights_float32/pre_trained_normal-nmnist_snn_300e.t7'
    model = NMNIST_model(T_BIN=15).to(device)

    # Dataset (TEST and Subset Loader)
    train_loader , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=15,dataset_path=dataset_path)
    UNTARGETED_loader = UNTARGETED_loader(target=target_dataset,num_images=num_images,batch_size=batch_size,T_BIN=15,dataset_path =dataset_path)
    num_class = 10

elif (target_dataset == "DVS128_Gesture"):

    encrypted_weight_path = '../pretrained_weights_float32/pretrained_DVS_csnn_128e_91a.t7'
    model = DVS128_model(T_BIN=40).to(device)

    # Dataset (TEST and Subset Loader)
    train_loader , test_loader = choose_dataset(target=target_dataset,batch_size=batch_size,T_BIN=60,dataset_path=dataset_path)
    UNTARGETED_loader = UNTARGETED_loader(target=target_dataset,num_images=num_images,batch_size=batch_size,T_BIN=60,dataset_path =dataset_path)
    num_class = 11

else:
    raise ValueError("Target dataset not recognized. (NMNIST/DVS128_Gesture)")


# quantized the model first after loading 
# checkpoint = torch.load(encrypted_weight_path,map_location=device)
# model.load_state_dict(checkpoint['net'])
# quantize_weights_nbits(model,quantized_bit)

#Model Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 3e-5)
num_epochs = 100

start = datetime.datetime.now()
print(datetime.datetime.now())

max_acc=0
Test_acc=[]
for epoch in range(num_epochs):
    avg_loss=0
    for batch_idx,(images, labels) in enumerate(train_loader):  #Each Batch # The enumerate() method adds a counter to an iterable
        images = torch.where(images > 0, torch.tensor(1.), torch.tensor(0.)) # For attack purpose, please remove for best ACC
        model.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        labels = labels.to(device)
        labels_onehot = F.one_hot(labels, num_class).float()  # AABBBCC -> 100, 100, 010, 010, 010, 001, 001        

        out_firing = model(images)

        # Loss function => MSE so one_hots for L2 distance error
        loss = criterion(out_firing, labels_onehot)
        loss.backward()
        optimizer.step() 
        
        avg_loss += loss.item()
    
    # Check acc
    print(f'Epoch Done: {epoch}')
    curr_acc = check_accuracy(test_loader,model).item()
    Test_acc +=[curr_acc]

    current_result = {}
    current_result["Epoch"] = epoch+1
    current_result["train_loss"] = avg_loss/len(train_loader)
    current_result["curr_acc"] = curr_acc
    current_result["max_acc"] = max_acc
    write_log(current_result, './fine_tuning_attack_'+target_dataset+'.json' )

    if curr_acc >= max_acc:
        max_acc = curr_acc
        print("State-of-the-art saving......")
        names = './fine_tuning_attack_'+target_dataset+'.t7'
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'acc_record': Test_acc,
        }
        torch.save(state, names)

end = datetime.datetime.now()
print(datetime.datetime.now())
print(f"Time = {end-start}")
