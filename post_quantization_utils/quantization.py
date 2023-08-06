import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size:torch, half_lvls):
        # ctx is a context object that can be used to stash information
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls

        # Clip weights to quantized range
        min_val = -ctx.half_lvls * ctx.step_size.item()
        max_val =  ctx.half_lvls * ctx.step_size.item()
        output = F.hardtanh(input, min_val=min_val, max_val=max_val)

        # Equivalent to Wl = round(Wfp/Δw_l)    
        output = torch.round(output / ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step_size
        return grad_input, None, None

quantize = _quantize_func.apply

class quan_Conv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        
        super(quan_Conv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        
        # INPUT: TArget bit quantization

        self.N_bits = 8

        ########################################### BIT_Quantization ##############################################
        ###########################################################################################################

        self.full_lvls = 2**self.N_bits              # 256                                                        2
        self.half_lvls = (self.full_lvls - 2) / 2    # 127  (2's complement overflow range: -128 <-> 127)         0
        
        # Initialize the step size = 1 (just to create a tensor scalar, 1 is not important)
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)

        # Initialize the real step size to Δw_l = max(Wl)/2^(N_bit-1) -1 for this convolutional layer
        self.__reset_stepsize__()

        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit {7,6,5,4,3,2,1,0} => {128, 64, 32, 16, 8, 4, 2, 1}
        # Unsqueeze (1D to 2D array)
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        # Two's complement b_w[0]=negative + b_w[i], Range: -2^(N_bit-1) to 2^(N_bit-1) -1 => {-128, 64, 32, 16, 8, 4, 2, 1}
        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative

        ########################################### BIT_Quantization ##############################################
        ###########################################################################################################

    def forward(self, input):
        # For Testing (Post-quantization, no longer need to update step_size) (No Gradient)
        if self.inf_with_weight:
            # Not even rounding
            return F.conv2d(input, self.weight * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
        # For Training (Gradient needed)
        else:                
            # Equation(2): Δw_l = max(Wl)/2^(N_bit-1) -1
            self.__reset_stepsize__()

            # Equation(3): Wl = round(Wfp/Δw_l).Δw_l
            weight_quan = quantize(self.weight, self.step_size, self.half_lvls) * self.step_size

            return F.conv2d(input, weight_quan, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        
    ########################################## UTIL Function ##################################################
    ###########################################################################################################

    def __reset_stepsize__(self): #Depends on the search space of weight, reset for every forward pass
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls
        
    def __reset_weight__(self):
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True

    ########################################## UTIL Function ##################################################
    ###########################################################################################################


class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)


        # INPUT: TArget bit quantization

        self.N_bits = 8

        ########################################### BIT_Quantization ##############################################
        ###########################################################################################################

        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2

        # Initialize the step size = 1 (just to create a tensor scalar, 1 is not important)
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)

        # Calculate the real step size for this Linear layer
        self.__reset_stepsize__()

        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit {7,6,5,4,3,2,1,0} => {128, 64, 32, 16, 8, 4, 2, 1}
        # Unsqueeze (1D to 2D array)
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place reverse

        ########################################### BIT_Quantization ##############################################
        ###########################################################################################################

    def forward(self, input):
        if self.inf_with_weight:  # No gradient is calculated
            return F.linear(input, self.weight * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            return F.linear(input, weight_quan, self.bias)


    ########################################## UTIL Function ##################################################
    ###########################################################################################################

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True

    ########################################## UTIL Function ##################################################
    ###########################################################################################################



def float2bin_gpu(f, fixed_exp, nbits=8):
    s = torch.sign(f)
    f = f * s
    s = (s * (-1) + 1.) * 0.5
    s= s.unsqueeze(-1)
    f = f/(2**fixed_exp)
    m = integer2bit(f - f % 1,num_bits = nbits-1)
    dtype = f.type()
    out = torch.cat([s, m], dim=-1).type(dtype)
    return out

def integer2bit(integer, num_bits=7):
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2

def bin2float_gpu(b, fixed_exp, nbits=8):
    dtype = torch.float32
    s = torch.index_select(b, -1, torch.arange(0, 1).to(device))
    m = torch.index_select(b, -1, torch.arange(1,nbits).to(device))
    out = ((-1) ** s).squeeze(-1).type(dtype)
    exponents = -torch.arange(-(nbits - 2.), 1.).to(device)
    exponents = exponents.repeat(b.shape[:-1] + (1,))
    e_decimal = torch.sum(m * 2 ** (exponents), dim=-1)
    out *= e_decimal * 2 ** fixed_exp
    return out

def bin_add_bit_error(b, bit_error_percent = 0):
    cond = torch.rand(b.shape).to(device) > bit_error_percent
    a = 1 - b
    out = cond * b + torch.logical_not(cond) * a
    return out

# evauluation: make sure the conversion functions work correctly
nbits = 12
f = copy.deepcopy(classifier.module.fc[0].weight.data)
weight_max = torch.max(f)
fixed_exp = torch.ceil(torch.log2(weight_max/(2**(nbits-1)-1))) + 1
# rec = bin2float_gpu(float2bin_gpu(f, fixed_exp, nbits),fixed_exp, nbits)
# if not torch.all((rec - f) <= 1 * 2**fixed_exp):
#     print(i, 'binary float conversion failed')
# if not torch.all(rec - f >= -1 * 2**fixed_exp):
#     print(i, 'binary float conversion failed')
            