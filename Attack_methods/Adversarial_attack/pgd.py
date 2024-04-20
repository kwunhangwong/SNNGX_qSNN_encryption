import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack


def G2S_Converter(x, grad): #output
    grad = F.normalize(grad, dim=(2,3,4))
    
    binary_mask = torch.bernoulli(grad)
    sign_extract = torch.sign(grad*binary_mask)

    result = overflow_binary_add(sign_extract,x)
    delta = result*binary_mask - x

    return result, delta

def overflow_binary_add(a, b):
    """
    Args:
        a (torch.Tensor): Input tensor, containing values 0 or 1.
        b (torch.Tensor): Input tensor, containing values -1, 0, or 1.
        
    Returns:
        torch.Tensor: Output tensor, containing values 0 or 1.
    """
    result = a + b
    c = torch.where(result < 0, 0, result)
    c = torch.where(c > 1, 1, c)
    return c


class PGD(Attack):
    def __init__(self, model, eps=8/255, steps=7):
        super().__init__("PGD", model)
        self.eps = eps      # Max budget

        self.steps = steps
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        ######## (Liang et al. 2021) ########
        loss = nn.MSELoss()
        #####################################
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            num_class = outputs.shape[-1]

            # Calculate loss
            if self.targeted:
                labels_onehot = F.one_hot(target_labels, num_class).float()    
                cost = -loss(outputs, labels_onehot)
            else:
                labels_onehot = F.one_hot(labels, num_class).float()  
                cost = loss(outputs, labels_onehot)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images,
                retain_graph=False, create_graph=False)[0]

            #### (liang et al. 2021) ####
            with torch.no_grad():

                # Restricted Spike Flipper (RSF), recommended gamma = 0.05, if all zeros gradient 
                if torch.equal(grad, torch.zeros_like(grad)):
                    gamma = 0.05
                    binary_mask = torch.bernoulli(torch.ones_like(grad)*gamma).to(torch.bool)
                    adv_images = torch.where(binary_mask, 1 - adv_images.detach(), adv_images.detach())

                # Gradient-to-Spike (G2S) Converter  
                else:
                    abs_grad = torch.abs(grad)
                    norm_grad = (abs_grad - torch.min(abs_grad)) / (torch.max(abs_grad) - torch.min(abs_grad))

                    binary_mask = torch.bernoulli(norm_grad)
                    sign_extract = torch.sign(grad*binary_mask)
                    
                    # print("Sparse no. of Zeros: ")
                    # print((binary_mask==0).sum().item())
                    # print("Total no.: ")
                    # print(len(binary_mask.view(-1)))

                    adv_images = overflow_binary_add(adv_images.detach(), sign_extract)
                    # delta = adv_result*binary_mask - adv_images.detach()

                perturbation =  (adv_images - images).view(-1)
                l2_perturbation = torch.norm(perturbation, p=2)/len(perturbation)
                if l2_perturbation >= self.eps:
                    print("attack fail")
                    break
                
                if self.targeted and torch.equal(torch.argmax(outputs, dim=1), target_labels):
                    print("targeted success")
                    return adv_images
                elif self.targeted == False and torch.all(torch.not_equal(torch.argmax(outputs, dim=1), labels)):
                    print("untargeted success")
                    return adv_images
    
        print("Gradient Vanishing")
        return adv_images

# adv_images = adv_images.detach() + self.alpha*grad.sign()
# delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
# adv_images = torch.clamp(images + delta, min=0, max=1).detach()