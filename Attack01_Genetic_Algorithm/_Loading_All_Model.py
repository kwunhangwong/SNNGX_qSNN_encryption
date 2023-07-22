import torch
import torch.nn as nn         
from torch.utils.data import DataLoader  
from binarization import *

# SNN Parameters
VTH = 0.3
DECAY = 0.3
alpha = 0.5

#Forward Model
class SNN_model(nn.Module):
    
    def __init__(self,input_size=2*34*34,num_classes=10,batch_size=64,
                 T_BIN = 15, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SNN_model,self).__init__()

        self.batch_size = batch_size
        self.T_BIN = T_BIN
        self.device = device
        
        self.fc1 = nn.Linear(input_size, 512 , bias = False)
        self.fc2 = nn.Linear(512, num_classes , bias = False)

    def forward(self,input):

        # Reseting Neurons
        h1_volt = h1_spike = h1_sumspike = torch.zeros(self.batch_size, 512, device=self.device)
        h2_volt = h2_spike = h2_sumspike = torch.zeros(self.batch_size, 10, device=self.device)

        for i in range(self.T_BIN): # Every single piece of t belongs to T

            x = input[i,:,:,:,:].reshape(self.batch_size, -1)

            h1_volt, h1_spike = mem_update(self.fc1, x, h1_volt, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike

            h2_volt, h2_spike = mem_update(self.fc2, h1_spike, h2_volt, h2_spike)
            h2_sumspike = h2_sumspike + h2_spike

        outputs = h2_sumspike / self.T_BIN
        return outputs
    

class ActivationFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(VTH).float()      # torch.gt(a,b) compare a and b : return 1/0 spike

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - VTH) < alpha/2    # lens = alpha/2
        return grad_input * temp.float() / alpha  # intensify spiking output (Wu et. 2018 w/o 2*len) 

act_fun = ActivationFun.apply

def mem_update(fc, x, volt, spike):
    volt = volt * DECAY * (1 - spike) + fc(x)
    spike = act_fun(volt)
    return volt, spike


class ANN_model(nn.Module):
    def __init__(self,num_classes=10):
        super(ANN_model, self).__init__()
        self.fc1 = BiLinear(784, 1000)  # 28*28=784 input features, 128 output features
        self.fc2 = BiLinear(1000, 512,binary_act=True)
        self.fc3 = BiLinear(512, num_classes,binary_act=True)  # 128 input features, 10 output features (one for each class)
        
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)  # apply ReLU activation to the first layer
        x = self.fc2(x)  # apply ReLU activation to the first layer
        x = self.fc3(x)  # output layer
        return x


#Model evaluation
def check_accuracy(loader, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    print("Checking on testing data")
    
    num_correct = 0
    num_sample = 0
    model.eval()  

    with torch.no_grad():   #no need to cal grad
        for image,label in loader:
            image= image.to(device)
            label= label.to(device)
            
            # T x N x 2312 => N x 2312
            out_firing = model(image)

            #64x10 output
            _ , prediction = out_firing.max(1)  #64x1 (value in 2nd dimension)
            num_correct += (prediction==label).sum()
            num_sample += prediction.size(0)  #64 (value in 1st dimension)
            
        print(f'Got {num_correct}/{num_sample} with accuracy {float(num_correct)/float(num_sample)*100:.2f}')
        
    model.train() #Set back to train mode
    return num_correct/num_sample    

