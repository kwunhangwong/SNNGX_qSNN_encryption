import torch
import torch.nn as nn         
import torch.nn.functional as F

# SNN Hyperparameters
VTH = 0.3
DECAY = 0.3
alpha = 0.5

# SNN surrogate gradient
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

# Forward Model (FC,CSNN)
class NMNIST_model(nn.Module):
    
    def __init__(self,input_size=2*34*34,num_classes=10,
                 T_BIN = 15, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(NMNIST_model,self).__init__()

        self.T_BIN = T_BIN
        self.device = device
        
        self.fc1 = nn.Linear(input_size, 512 , bias = False)
        self.fc2 = nn.Linear(512, num_classes , bias = False)

    def forward(self,input):

        _, batch_size, _, _, _ = input.shape

        # Reseting Neurons
        h1_volt = h1_spike = h1_sumspike = torch.zeros(batch_size, 512, device=self.device)
        h2_volt = h2_spike = h2_sumspike = torch.zeros(batch_size, 10, device=self.device)

        for i in range(self.T_BIN): 

            x = input[i,:,:,:,:].reshape(batch_size, -1)

            h1_volt, h1_spike = mem_update(self.fc1, x, h1_volt, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike

            h2_volt, h2_spike = mem_update(self.fc2, h1_spike, h2_volt, h2_spike)
            h2_sumspike = h2_sumspike + h2_spike

            torch.cuda.empty_cache()
            del x

        outputs = h2_sumspike / self.T_BIN

        torch.cuda.empty_cache()
        del h1_volt, h1_spike, h2_volt, h2_spike, h2_sumspike
        return outputs


class DVS128_model(nn.Module):

    def __init__(self,in_channels=2,num_classes=11,
                 T_BIN = 15, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(DVS128_model,self).__init__()

        self.T_BIN = T_BIN
        self.device = device

        self.pool  = nn.MaxPool2d(2,2)

        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.conv1 = nn.Conv2d( 64 ,  128,      kernel_size=3, stride=1, padding=1, bias=False)        
        self.conv2 = nn.Conv2d( 128 , 128,      kernel_size=3, stride=1, padding=1, bias=False)       
        self.conv3 = nn.Conv2d( 128 , 256,      kernel_size=3, stride=1, padding=1, bias=False)       #

        self.fc1   = nn.Linear(4 * 4 * 256, 1024, bias = False)  # 4096*1024 
        self.fc2   = nn.Linear(1024, num_classes, bias = False) 

    def forward(self,input):

        _, batch_size, _, _, _ = input.shape

        # Reseting Neurons
        c0_mem = c0_spike = torch.zeros(batch_size, 64, 32, 32,device=self.device)
        c1_mem = c1_spike = torch.zeros(batch_size, 128, 16, 16, device=self.device) 
        c2_mem = c2_spike = torch.zeros(batch_size, 128, 8, 8, device=self.device)
        c3_mem = c3_spike = torch.zeros(batch_size, 256, 4, 4,   device=self.device)

        h1_mem = h1_spike = torch.zeros(batch_size, 1024, device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, 11, device=self.device)

        for i in range(self.T_BIN): 
            x = input[i,:,:,:,:].to(self.device)

            c0_mem, c0_spike = mem_update(self.conv0, x, c0_mem, c0_spike)
            p0_spike = self.pool(c0_spike)
            
            c1_mem, c1_spike = mem_update(self.conv1, p0_spike, c1_mem, c1_spike) 
            p1_spike = self.pool(c1_spike) 

            c2_mem, c2_spike = mem_update(self.conv2, p1_spike, c2_mem, c2_spike) 
            p2_spike = self.pool(c2_spike) 

            c3_mem, c3_spike = mem_update(self.conv3, p2_spike, c3_mem, c3_spike) 
            x = c3_spike.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
            
            torch.cuda.empty_cache()
            del x

        outputs = h2_sumspike / self.T_BIN

        torch.cuda.empty_cache()
        del c0_mem, c0_spike, c1_mem, c1_spike, c2_mem, c2_spike, c3_mem, c3_spike, h1_mem, h1_spike, h2_mem, h2_spike, h2_sumspike
        return outputs


# Model evaluation
def check_accuracy(loader, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    print("Checking on testing data")
    num_correct = 0
    num_sample = 0
    model.eval()  
    with torch.no_grad():   #no need to cal grad
        for image,label in loader:
            image= image.to(device)
            label= label.to(device)

            out_firing = model(image)
            _ , prediction = out_firing.max(1)  
            num_correct += (prediction==label).sum()
            num_sample += prediction.size(0) 

        print(f'Got {num_correct}/{num_sample} with accuracy {float(num_correct)/float(num_sample)*100:.2f}')
    model.train() 
    return num_correct/num_sample    
