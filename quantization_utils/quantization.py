import torch 
import torch.nn as nn

def float2bin(f, fixed_exp, nbits=8):
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

def bin2float(b, fixed_exp, nbits=8, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    dtype = torch.float32
    s = torch.index_select(b, -1, torch.arange(0, 1).to(device))
    m = torch.index_select(b, -1, torch.arange(1,nbits).to(device))
    out = ((-1) ** s).squeeze(-1).type(dtype)
    exponents = -torch.arange(-(nbits - 2.), 1.).to(device)
    exponents = exponents.repeat(b.shape[:-1] + (1,))
    e_decimal = torch.sum(m * 2 ** (exponents), dim=-1)
    out *= e_decimal * 2 ** fixed_exp
    return out

def quantize_weights_nbits(model, nbits):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            print(f"The current layer is: {name}: ")
            weight = module.weight.data
            # step size
            weight_max = torch.max(weight)
            fixed_exp = torch.ceil(torch.log2(weight_max/(2**(nbits-1)-1))) 

            # quantize to binary, and back to floating pt
            binary = float2bin(weight,fixed_exp,nbits).to(torch.int8)
            quantized_f = bin2float(binary,fixed_exp,nbits)

            # update weight
            module.weight.data = quantized_f
            print(f"finished quantized {name} weights to {nbits} BITs")
    return None

def quantize_to_binary(weight, nbits):

    # get weights exponential 
    weight_max = torch.max(weight)
    fixed_exp = torch.ceil(torch.log2(weight_max/(2**(nbits-1)-1))) 

    # print(f"quantize_to_binary: {fixed_exp}")

    # quantize to binary
    binary = float2bin(weight,fixed_exp,nbits).to(torch.int8)

    # return the shape of the binary vector
    bit_shape = binary.shape

    # number of weights * nbits
    all_binary_tensor = binary.view(-1).detach().clone()

    # change the bin tensor from {0,1} to {-1,+1}
    output_tensor = 2. * all_binary_tensor - 1.  

    return output_tensor, bit_shape    # output_tensor = [-1., +1., -1., -1., -1.,....], bit_shape = (Wm x Wn x nbits)


def binary_to_weight32(weight, nbits, input_tensor, bit_shape):

    # get weights exponential (weight = layer.weight.data)
    weight_max = torch.max(weight)
    fixed_exp = torch.ceil(torch.log2(weight_max/(2**(nbits-1)-1))) 

    # print(f"binary_to_weight32: {fixed_exp}")

    # change the bin tensor from {-1,+1} to {0,1}
    input_tensor = ((input_tensor + 1)/2).to(torch.int8)
    
    # change to output shape 
    output_tensor = input_tensor.view(bit_shape)     

    # and back to floating pt representation
    quantized_f = bin2float(output_tensor,fixed_exp,nbits)

    return quantized_f
