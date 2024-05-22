import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('../quantization_utils')

from quantization import *

def Random_flipping_all_Layers(num:float, model:nn.Module, qbits:int):

    dim_storage = []
    BIT_array = []
    list_sep = []

    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            
            weight = module.weight.data
            weight1d, bit_shape = quantize_to_binary(weight, qbits)
            dim_storage += [bit_shape]

            # All layer weights matrix collapse to one 1d array
            BIT_array += weight1d.tolist()

            # TAIL positsion of curr layer (not HEAD of next layer)   
            list_sep += [len(BIT_array)]
            
    new_BIT = np.array(BIT_array).astype(np.float32)
    # raw_BIT = new_BIT.copy()
    # new_BIT = Only_kBits(new_BIT,qbits)

    # Generate an array of integers from 0 to 1,000,000 BITS
    integers = np.arange(len(new_BIT))
    print(f"This model has: {len(new_BIT)} BITS in total")

    # Shuffle the array in place
    np.random.shuffle(integers)

    # Select the first (input) elements of the shuffled array
    num = int(num*len(new_BIT))
    random_pos = integers[:num]
    new_BIT[random_pos] *= -1

    # new_BIT = Only_kBits_Recov(raw_BIT,new_BIT,qbits)

    head = 0
    pos = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

            weight = module.weight.data
            # Recover 1d array into some layers of 1d array
            tail = list_sep[pos]
            layer_weight = torch.from_numpy(new_BIT[head:tail]).to(torch.device('cuda'))

            # White-Box: Update weights
            quantized_f = binary_to_weight32(weight, qbits, layer_weight, dim_storage[pos])
            module.weight.data = quantized_f
            
            head = tail
            pos+=1
    return model


def Random_flipping_single_layer(num:float, model:nn.Module, qbits:int, layer_idx:int):

    layer_cnt = 0 
    target_layer = None
    for child in model.children():
        if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
            if (layer_cnt==layer_idx):
                target_layer = child
            layer_cnt += 1

    if target_layer is not None:
        print("The current layer is:", target_layer)

        # Transform to array
        weight = target_layer.weight.data
        weight1d, bit_shape = quantize_to_binary(weight, qbits)
        new_BIT = weight1d.to(torch.device('cpu')).numpy().astype(np.float32)

        # raw_BIT = new_BIT.copy()
        # new_BIT = Only_kBits(new_BIT,qbits)

        # Generate an array of integers from 0 to 1,000,000 BITS
        integers = np.arange(len(new_BIT))
        print(f"This layer has: {len(new_BIT)} BITS in total")

        # Shuffle the array in place
        np.random.shuffle(integers)

        # Select the first (input) elements of the shuffled array
        num = int(num*len(new_BIT))
        random_pos = integers[:num]
        new_BIT[random_pos] *= -1

        # White-Box: Update weights
        # new_BIT = Only_kBits_Recov(raw_BIT,new_BIT,qbits)

        weight1d = torch.from_numpy(new_BIT).to(torch.device('cuda'))
        quantized_f = binary_to_weight32(weight, qbits, weight1d, bit_shape)
        target_layer.weight.data = quantized_f

        return model

    else:
        print("No layer found in the model.")


def Only_kBits(BIT_array:np,qbits:int):
    pos_array = np.arange(len(BIT_array))
    
    # Only Sign bit 0,4,8
    selected_elements = BIT_array[(pos_array % qbits == 0)]

    # Only Largest bit
    # selected_elements = BIT_array[(pos_array % qbits == 1)]

    # Most critical 2 bits (Sign & Largest)
    # selected_elements = BIT_array[np.logical_or((pos_array % qbits == 0),(pos_array % qbits == 1))]

    return selected_elements


def Only_kBits_Recov(raw_Arr,r_adv_Arr:np,qbits): 

    pos_array = np.arange(len(raw_Arr))
    adv_Array = raw_Arr.copy()

    # Only Sign bit 0,4,8
    adv_Array[(pos_array % qbits == 0)] = r_adv_Arr

    # Only Largest bit
    # adv_Array[(pos_array % self.qbits == 1)] = r_adv_Arr

    # Most critical 2 bits (Sign & Largest)
    # adv_Array[np.logical_or((pos_array % self.qbits == 0),(pos_array % self.qbits == 1))] = r_adv_Arr

    return adv_Array