import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('../quantization_utils')

from quantization import *

def Random_flipping_all_Layers(num:int, model:nn.Module, qbits:int):

    dim_storage = []
    BIT_array = []
    list_sep = []

    for weight in model.parameters():

        # Transform to array (3D matrix => 1D matrix)
        weight = weight.data
        weight1d, bit_shape = quantize_to_binary(weight, qbits)
        dim_storage += [bit_shape]

        # All layer weights matrix collapse to one 1d array
        BIT_array += weight1d.tolist()

        # TAIL positsion of curr layer (not HEAD of next layer)   
        list_sep += [len(BIT_array)]

    new_BIT = np.array(BIT_array).astype(np.float32)

    # Generate an array of integers from 0 to 1,000,000 BITS
    integers = np.arange(len(new_BIT))
    print(f"This model has: {len(new_BIT)} BITS in total")

    # Shuffle the array in place
    np.random.shuffle(integers)

    # Select the first (input) elements of the shuffled array
    random_pos = integers[:num]
    new_BIT[random_pos] *= -1
    
    head = 0
    for layer_idx, weight in enumerate(model.parameters()):

            # Recover 1d array into some layers of 1d array
            tail = list_sep[layer_idx]
            layer_weight = torch.from_numpy(new_BIT[head:tail])

            # White-Box: Update weights
            quantized_f = binary_to_weight32(weight, qbits, layer_weight, dim_storage[layer_idx])
            weight.data = quantized_f
            head = tail

    return model


def Random_flipping_single_layer(num:int, model:nn.Module, qbits:int, layer_type:nn.Module, layer_idx:int):

    layer_cnt = 0 
    target_layer = None
    for child in model.children():
        if isinstance(child, layer_type):
            layer_cnt += 1
            if (layer_cnt==layer_idx):
                target_layer = child

    if target_layer is not None:
        print("The current layer is:", target_layer)

        # Transform to array
        weight = target_layer.weight.data
        weight1d, bit_shape = quantize_to_binary(weight, qbits)
        new_BIT = weight1d.numpy().astype(np.float32)

        # Generate an array of integers from 0 to 1,000,000 BITS
        integers = np.arange(len(new_BIT))
        print(f"This layer has: {len(new_BIT)} BITS in total")

        # Shuffle the array in place
        np.random.shuffle(integers)

        # Select the first (input) elements of the shuffled array
        random_pos = integers[:num]
        new_BIT[random_pos] *= -1

        # White-Box: Update weights
        weight1d = torch.from_numpy(new_BIT)
        quantized_f = binary_to_weight32(weight, qbits, weight1d, bit_shape)
        target_layer.weight.data = quantized_f

        return model

    else:
        print("No layer found in the model.")