import torch
import torch.nn as nn
import torch.nn.functional as F 
import operator

import sys

sys.path.append('../quantization_utils')
from quantization import *

class FGSM_Untargeted_BIT_flip():

    def __init__(self, model:nn.Module, k_top=5, expected_bits=20, iteration=10, qbits=8,is_ANN=False):
        
        # import Model 
        self.model = model
        self.is_ANN = is_ANN
        self.model.eval()

        # Finding top k vulnerable BITS, suggested expected bits = k_top*4 in 10 iter
        self.k_top = k_top
        self.expected_bits = expected_bits
        self.qbits = qbits

        # FGSM-variable
        self.iteration = iteration  # On vulnerable layer
        self.loss = 0

        # Log File
        self.loss_dict = {}

    def pos_to_qpos(self, float32_pos, dtype):
        sequence_quantized = torch.Tensor([])
        for i in range(len(float32_pos)):

            ##### Flip qbits #####
            # new_pos_begin = (float32_pos[i]-1)*self.qbits 
            # end_pos_begin = (float32_pos[i])*self.qbits

            ##### Dont include sign bit ##### because our of quantized representation 
            # new_pos_begin = (float32_pos[i]-1)* self.qbits +1  # No sign bit
            # end_pos_begin = (float32_pos[i])* self.qbits

            ##### Flip only sign bit #####
            # new_pos_begin = (float32_pos[i]-1)*self.qbits     # No sign bit
            # end_pos_begin = (float32_pos[i]-1)*self.qbits +1  # only sign bits

            ##### Flip only 2nd bit #####
            new_pos_begin = (float32_pos[i]-1)*self.qbits +1
            end_pos_begin = (float32_pos[i]-1)*self.qbits +2          

            sequence = torch.arange(new_pos_begin, end_pos_begin)
            sequence_quantized = torch.cat((sequence_quantized, sequence), dim=0)
        return sequence_quantized.to(dtype)

    def flip_bit(self,layer):
        # Feeding in one layer's weight assume
        print(f"flipping:{layer}")

        # only largest gradient: return k largest grad, and k position tensor
        w_grad_topk, w_idx_topk = layer.weight.grad.detach().abs().view(-1).topk(self.k_top)
        w_grad_topk = layer.weight.grad.detach().view(-1)[w_idx_topk]              # Extract signed gradient value

        # Ensure Max gradient is not zero
        if w_grad_topk.abs().max().item() == 0: 
            print(f"The max grad is zero, iteration:{self.iteration}, layer:{layer}")

        # BITS finding with their signs
        b_grad_topk_sign = w_grad_topk.sign()              # return -1., 0., +1.  shape = topk
        b_grad_topk_sign[b_grad_topk_sign.eq(0)]=1         # return -1., +1.      shape = topk

        # Weight with largest gradient position
        model_weight= layer.weight.data.detach().clone()          

        # model_weight_topk: Bits array with largest gradient
        weight1d, bit_shape = quantize_to_binary(model_weight, self.qbits)        # weight quantization: +1,-1,+1.......
        q_w_idx_topk = self.pos_to_qpos(w_idx_topk, w_idx_topk.dtype) # Length may change depends on flip q/q-1/1 bits per float32
        model_weight_topk = ((weight1d[q_w_idx_topk] + 1) * 0.5).to(torch.int)    #                    model_weight_topk: 1, 0, 1, 0  (0+ 1- 0+ 1-)

        # b_grad_topk_sign: model_weight_topk's gradient sign
        b_grad_topk_sign = ((b_grad_topk_sign + 1) * 0.5).to(torch.int)   # return  0, +1 (int type)    b_grad_topk_sign: 1, 1, 0, 0  (1+ 0- 0- 1+)
                                                                                                                                     #(0+ 1- 1- 0+)
        expand_constant = int(len(model_weight_topk)/len(b_grad_topk_sign))    # need to expand b_grad_topk_sign to multiple of q/q-1/1 according the length of model_weight_topk
        b_grad_topk_sign = torch.repeat_interleave(b_grad_topk_sign,expand_constant)

        # overflow mask: grad_mask
        grad_mask = b_grad_topk_sign ^ model_weight_topk                  #                                    grad_mask: 0, 1, 1, 0  (0+, )

        # XOR operator '^' on mask m
        # weight1d[q_w_idx_topk] = (((grad_mask ^ model_weight_topk)-1)*-1).to(torch.float)  #                       final_weight: 1, 1, 0, 0
        weight1d[q_w_idx_topk] = (grad_mask ^ model_weight_topk).to(torch.float) 
        weight1d[weight1d.eq(0)]= -1                                              # return from 0.,1. to -1., +1.   
        # weight1d[(w_idx_topk-1)*self.qbits] *= -1
        model_weight = binary_to_weight32(model_weight, self.qbits, weight1d, bit_shape)

        # Update_weight in this layer 
        if isinstance(layer, nn.Conv2d): # 4D tensor: in x out x kernel(3x3) 
            layer.weight.data = model_weight
        elif isinstance(layer, nn.Linear): # 2D tensor: in x out
            layer.weight.data = model_weight
        else: 
            print("Error: Weights shape update !!")
        
        # Return raw tensor
        return weight1d  #attack_weight 1D tensor

    def progressive_bit_search(self, data, target):

        if self.is_ANN:
            criterion = nn.CrossEntropyLoss()
            labels_onehot = target

        else: 
            criterion = nn.MSELoss()
            labels_onehot = F.one_hot(target, 10).float()

        out_firing = self.model(data)   
        self.loss = criterion(out_firing, labels_onehot)

        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    layer.weight.grad.data.zero_()

        self.loss.backward()

        # Cross-layer search attack
        loss_max = self.loss.item()
        max_loss_module = ''
        while loss_max <= self.loss.item():

            # iterate all the quantized conv and linear layer
            for name, layer in self.model.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    
                    # BIT Flip with largest gradient
                    clean_weight = layer.weight.data.clone()

                    self.flip_bit(layer)

                    # Change the weight to attacked weight and get loss
                    out_firing = self.model(data) 
                    self.loss_dict[name] = criterion(out_firing, labels_onehot).item()

                    # change the weight back to the clean weight (zero grad)
                    layer.weight.data = (clean_weight)

            max_loss_module = max(self.loss_dict.items(), key=operator.itemgetter(1))[0]
            loss_max = self.loss_dict[max_loss_module]          #   max_loss_module is a "name" e.g. nn.Linear

        #check_accuracy(test_loader,self.model)
        # Targeting at maximum loss layer
        for name, layer in self.model.named_modules():
            if (name == max_loss_module):

                clean_weight = layer.weight.data
                clean_weight, _ = quantize_to_binary(clean_weight, self.qbits) 

                for i in range(self.iteration):

                    with torch.set_grad_enabled(True):
                        # layer.weight.grad.data.zero_()
                        self.model.zero_grad()

                        out_firing = self.model(data)     
                        self.loss = criterion(out_firing, labels_onehot)
                        self.loss.backward()

                        print(self.loss.item())
                        self.loss_dict[max_loss_module] = self.loss.item()
                        
                        attack_weight = self.flip_bit(layer)
                        self.k_top += 1

                        print(f"Bits flipped: {(clean_weight != attack_weight).sum()}")

                        if (clean_weight != attack_weight).sum() >= self.expected_bits:
                            print(f"iteration :{i}")
                            break

        print(self.loss_dict)
        return (clean_weight != attack_weight).sum() , len(clean_weight)
