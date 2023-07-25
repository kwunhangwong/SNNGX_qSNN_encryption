import torch
import torch.nn as nn
import torch.nn.functional as F 
import operator

class FGSM_Untargeted_BIT_flip():

    def __init__(self, model:nn.Module, k_top=5, expected_bits=20, iteration=10):
        
        # import Model 
        self.model = model
        self.model.eval()

        # Finding top k vulnerable BITS, suggested expected bits = k_top*4 in 10 iter
        self.k_top = k_top
        self.expected_bits = expected_bits

        # FGSM-variable
        self.iteration = iteration  # On vulnerable layer
        self.loss = 0

        # Log File
        self.loss_dict = {}

        # SNN model hyperparameters
        # self.VTH, self.alpha = 0.3, 0.5 

    def flip_bit(self,layer):

        print(f"flipping:{layer}")

        # Find largest gradient: Floating point 32 value
        w_grad_topk, w_idx_topk = layer.weight.grad.detach().abs().view(-1).topk(self.k_top)
        w_grad_topk = layer.weight.grad.detach().view(-1)[w_idx_topk]              # Extract signed gradient value

        # Ensure Max gradient is not zero
        if w_grad_topk.abs().max().item() == 0:  # ensure the max grad is not zero
            print(f"The max grad is zero, iteration:{self.iteration}, layer:{layer}")

        # BITS finding with their signs
        b_grad_topk_sign = w_grad_topk.sign()              # return -1., 0., +1.  shape = topk
        b_grad_topk_sign[b_grad_topk_sign.eq(0)]=1         # return -1., +1.      shape = topk

        # Weight with largest gradient position
        model_weight= layer.weight.data.detach().clone().view(-1)                  # return  -1., +1.  shape = len(weight)
        model_weight_topk = ((model_weight[w_idx_topk] + 1) * 0.5).to(torch.int)   # return   0,   1   shape = topk

        # Finding the overflow mask
        b_grad_topk_sign = ((b_grad_topk_sign + 1) * 0.5).to(torch.int)   # return  0, +1 (int type)    b_grad_topk_sign: 1, 1, 0, 0 
        grad_mask = b_grad_topk_sign ^ model_weight_topk                  # return  0, 1, 1, 0         model_weight_topk: 1, 0, 1, 0

        # XOR operator '^' on mask m
        model_weight[w_idx_topk] = (grad_mask ^ model_weight_topk).to(torch.float)                              # return  1, 1, 0, 0
        model_weight[model_weight.eq(0)]= -1                              # return -1., +1.  shape = len(weight)

        # Update_weight in this layer 
        if isinstance(layer, nn.Conv2d): # 4D tensor: in x out x kernel(3x3) 
            layer.weight.data = model_weight.view(layer.weight.shape[0],layer.weight.shape[1],layer.weight.shape[2],layer.weight.shape[3])
        elif isinstance(layer, nn.Linear): # 2D tensor: in x out
            layer.weight.data = model_weight.view(layer.weight.shape[0],layer.weight.shape[1])
        else: 
            print("Error: Weights shape update !!")
        
        # Return raw tensor
        return model_weight  #attack_weight 1D tensor

    #   I DONT UNDERSTAND WHY he used eval here, how to calculate gradient?
    def progressive_bit_search(self, data, target):

        criterion = nn.MSELoss()
        labels_onehot = F.one_hot(target, 10).float()

        # Gradient calculate w.r.t. target label
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
                    clean_weight = layer.weight.data.detach().clone()

                    self.flip_bit(layer)

                    # Change the weight to attacked weight and get loss
                    out_firing = self.model(data) 
                    self.loss_dict[name] = criterion(out_firing, labels_onehot).item()

                    # change the weight back to the clean weight (zero grad)
                    #layer.weight.data = nn.Parameter(clean_weight)
                    layer.weight.data = (clean_weight)

             # after going through all the layer, now we find the layer with max loss
             # itemgetter(1) is used to extract the second element (i.e., the loss value) from 
             # each (key, value) pair in the dictionary, so that max() can compare the loss values 
             # and return the (key, value) pair with the highest loss value.
            max_loss_module = max(self.loss_dict.items(), key=operator.itemgetter(1))[0]
            loss_max = self.loss_dict[max_loss_module]          #   max_loss_module is a "name" e.g. nn.Linear

        #check_accuracy(test_loader,self.model)
        # Targeting at maximum loss layer
        for name, layer in self.model.named_modules():
            if name == max_loss_module:

                clean_weight = layer.weight.data.clone().view(-1)
                for i in range(self.iteration):

                    layer.weight.grad.data.zero_()

                    out_firing = self.model(data)        
                    self.loss = criterion(out_firing, labels_onehot)
                    self.loss.backward()

                    print(self.loss.item())
                    self.loss_dict[max_loss_module] = self.loss.item()
                    
                    attack_weight = self.flip_bit(layer)

                    if (clean_weight != attack_weight).sum() >= self.expected_bits:
                        print(f"iteration :{i}")
                        break

        print(self.loss_dict)
        return (clean_weight != attack_weight).sum() , len(clean_weight)
    