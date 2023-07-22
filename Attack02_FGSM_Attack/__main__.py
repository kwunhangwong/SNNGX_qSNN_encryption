import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader  


from _Loading_All_Model import SNN_model, ActivationFun, act_fun, mem_update, check_accuracy, VTH, DECAY, alpha
from Fast_Grad_Sign_BITver import FGSM_Untargeted_BIT_flip

import datetime


# Constant
batch_size = 64

# MODEL 1
snnn = SNN_model(batch_size=batch_size)
checkpoint = torch.load('../pretrained_models/pre_trained_binarised-nmnist_snn_300e.t7')
snnn.load_state_dict(checkpoint['net'])

# Dataset (UNTARGETED and TEST Loader)
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size, n_time_bins=15)])

trainset = tonic.datasets.NMNIST(save_to="../dataset",transform=frame_transform, train=True)    
testset = tonic.datasets.NMNIST(save_to="../dataset", transform=frame_transform, train=False)

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
    shuffle = False,
    drop_last=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Before Untargeted Attack: {check_accuracy(test_loader,snnn)*100}% Accuracy")
start = datetime.datetime.now()
print(datetime.datetime.now())

# Using Train_loader for FGSM attacks
for _, (data, target) in enumerate(train_loader):
    # Get First Batch of train_loader
    data = data.float().to(device)
    target = target.to(device)

    # Override the target to prevent label leaking
    _, target = snnn(data).max(1)
    break

# Attack finding minimum 10 bitflip
FGSM_attack = FGSM_Untargeted_BIT_flip(snnn, k_top=10,expected_bits=20, iteration=10)
bit_flipped, numlayer_weight = FGSM_attack.progressive_bit_search(data,target)
print(f'Total BITS flipped: {bit_flipped}')    
end = datetime.datetime.now()
print(datetime.datetime.now())

print(f"After Untargeted Attack: {check_accuracy(test_loader,snnn)*100:.2f}% Accuracy, BITS Flipped:{bit_flipped} out of {numlayer_weight}")
print(f"Time = {end-start}")

