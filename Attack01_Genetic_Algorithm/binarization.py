import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# Binarized to '+1' and '-1'
class BinaryQuantizeN(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)  # return -1, 0, +1
        out[out.eq(0)]=1         # return -1, +1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0  #Computes input > .1. element-wise.
        grad_input[input[0].lt(-1)] = 0  #Computes input < .-1. element-wise.
        return grad_input
    
# Binarized to '+1' and '-1' and '0'
class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input) 
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input
    

# Binarized to '+1' and '0' (For Activation)
class BinaryQuantizeZ(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out.le(0)] = 0           #Computes input <= .0. element-wise.
        return out

    @staticmethod
    def backward(ctx, grad_output): 
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input

# Binary_act = False, Bias = False, bw will not apply BinaryQuantizeN
class BiLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, binary_act=False):
        super(BiLinear, self).__init__(in_features, out_features, bias)
        self.binary_act = binary_act
        self.have_bias = bias
        self.output_ = None

    def forward(self, input): 
        # Weights and Activation (model.layername.weight)
        bw = self.weight
        ba = input

        # Binarized Weights (DISABLED!!!)
        # bw = BinaryQuantizeN().apply(bw)
        # Binarized Activation
        if self.binary_act:
            ba = BinaryQuantizeZ().apply(ba)
        # Add Biase
        if self.have_bias == True:
            output = F.linear(ba, bw, self.bias) 
        else:
            output = F.linear(ba, bw)

        # Returning y = xA + b
        self.output_ = output
        return output
    

# Binarized to '+1' and '-1'
class BinaryQuantizeN(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)  # return -1, 0, +1
        out[out.eq(0)]=1         # return -1, +1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0  #Computes input > .1. element-wise.
        grad_input[input[0].lt(-1)] = 0  #Computes input < .-1. element-wise.
        return grad_input
    

# Binarized to '+1' and '0' (For Activation)
class BinaryQuantizeZ(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out.le(0)] = 0           #Computes input <= .0. element-wise.
        return out

    @staticmethod
    def backward(ctx, grad_output): 
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input

# Binary_act = False, Bias = False, bw will not apply BinaryQuantizeN
class BiLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, binary_act=False):
        super(BiLinear, self).__init__(in_features, out_features, bias)
        self.binary_act = binary_act
        self.have_bias = bias
        self.output_ = None

    def forward(self, input): 
        # Weights and Activation (model.layername.weight)
        bw = self.weight
        ba = input

        if self.binary_act:
            ba = BinaryQuantizeZ().apply(ba)
        # Add Biase
        if self.have_bias == True:
            output = F.linear(ba, bw, self.bias) 
        else:
            output = F.linear(ba, bw)

        # Returning y = xA + b
        self.output_ = output
        return output
    
class BiConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                    padding, dilation=1, groups=1,
                    bias=False, padding_mode='zeros',binary_act=False):
        
        super(BiConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        self.binary_act = binary_act
        

    def forward(self, input):
        # Weights and Activation (model.layername.weight)
        bw = self.weight
        ba = input

        if self.binary_act:
            ba = BinaryQuantizeZ.apply(ba)

        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
                  
            