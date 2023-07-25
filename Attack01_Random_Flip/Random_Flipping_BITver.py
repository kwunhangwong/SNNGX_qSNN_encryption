import torch
import torch.nn as nn
import numpy as np

def Random_flipping_all_Layers(num:int, model:torch.nn.Module):

    BIT_array = []
    list_sep = []
    list_cont = 0

    for weight in model.parameters():

            # Transform to array
            weight1d = (weight.reshape(-1).detach()).tolist()

            # All layer weights matrix collapse to one 1d array
            BIT_array += weight1d

            # TAIL positsion of curr layer (not HEAD of next layer)   
            list_cont += len(weight1d)
            list_sep += [list_cont]
    
    new_BIT = np.array(BIT_array).astype(np.float32)

    # Generate an array of integers from 0 to 1,000,000 BITS
    integers = np.arange(len(new_BIT))

    # Shuffle the array in place
    np.random.shuffle(integers)

    # Select the first (input) elements of the shuffled array
    random_pos = integers[:num]
    new_BIT[random_pos] *= -1
    
    head = 0
    for layer_idx, weight in enumerate(model.parameters()):

            # Recover 1d array to 2d matrix
            tail = list_sep[layer_idx]

            layer_weight = new_BIT[head:tail]
            layer_weight = torch.from_numpy(layer_weight)
            
            # White-Box: Update weights
            weight.data = torch.nn.Parameter(layer_weight.reshape(weight.shape[0],weight.shape[1]))
            head = tail

    return model


def Random_flipping_single_layer(num:int, model:nn.Module, layer_type:nn.Module, layer_idx:int):

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
        weight = target_layer.weight
        weight1d = (weight.reshape(-1).detach()).tolist()
        new_BIT = np.array(weight1d).astype(np.float32)

        # Generate an array of integers from 0 to 1,000,000 BITS
        integers = np.arange(len(new_BIT))

        # Shuffle the array in place
        np.random.shuffle(integers)

        # Select the first (input) elements of the shuffled array
        random_pos = integers[:num]
        new_BIT[random_pos] *= -1

        # White-Box: Update weights
        weight1d = torch.from_numpy(new_BIT)
        target_layer.weight.data = torch.nn.Parameter(weight1d.reshape(target_layer.weight.shape[0],target_layer.weight.shape[1]))

        return model

    else:
        print("No layer found in the model.")