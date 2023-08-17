import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader

import sys

sys.path.append('../quantization_utils')
from quantization import *

class GA_BIT_flip_Untargeted: #Untargeted: Gen 25, eps = 5000, 100/10000
    
    def __init__(self, model:nn.Module, UNTARGETED_loader:DataLoader,   # Accepted input: only Single 4-D neuromorphic Data
                 epsil=5000, n_generations=25, population_size=100,      # epsil = L1/Hamming Distance bound
                 retain_best=0.6, mutate_chance=0.05, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 BITS_by_layer:bool=False, layer_type:nn.Module=None, layer_idx:int=None, qbits:int=8):  
        
        # Initialization
        self.model = model.to(device)
        self.epsil = epsil
        self.device = device

        # Population (must be even number)
        self.n_generations = n_generations
        self.population_size = population_size
        
        # Evolve
        self.retain_best = retain_best
        
        # Mutation
        self.mutate_chance = mutate_chance

        # Weights (list_sep => (By_layer)type:nn.Module or (All_layer)type:np)
        self.BITS_by_layer = BITS_by_layer
        self.qbits = qbits
        self.BIT_array, self.dim_storage, self.list_sep, self.target_layer = BITS_To_1D(model, layer_type=layer_type, 
                                                                                layer_idx=layer_idx, qbits=self.qbits,
                                                                                BITS_by_layer=BITS_by_layer)

        # Untargeted Attack
        self.UNTARGETED_loader = UNTARGETED_loader
    

    def init_Pop(self):
        
        adam_and_eve = []
        for i in range(self.population_size):

            # Part Two (All flipped weight)
            BIT_array_adv = self.BIT_array.copy()
            BIT_array_adv *= -1
            BIT_array_adv = self.reduced_mutation(BIT_array_adv)
            adam_and_eve.append(BIT_array_adv)

        return np.array(adam_and_eve)
            
    
    def reduced_mutation(self, BIT_array_adv:np): #np.type

        # mask equal*1, not equal*-1*0.05(chance)
        mutate_mask = np.equal(self.BIT_array,BIT_array_adv).astype(int)   #1111110000000
        flipping_position = np.random.binomial(1, self.mutate_chance,len(BIT_array_adv)) * -1  
        mutate_mask[mutate_mask==0] = flipping_position[mutate_mask==0]    #111111000-100-1
        mutate_mask[mutate_mask==0] = 1                                    #111111111-111-1
        # BIT Flip
        BIT_array_adv = BIT_array_adv*mutate_mask

        return BIT_array_adv #np.type


    def updateWeight(self, new_BIT:np):
        
        if (self.BITS_by_layer):

            # White-Box: Update weights
            weight = self.target_layer.weight.data

            layer_weight = new_BIT.astype(np.float32)
            weight1d = torch.from_numpy(layer_weight)

            quantized_f = binary_to_weight32(weight, self.qbits, weight1d, self.dim_storage)

            if isinstance(self.target_layer, nn.Linear) or isinstance(self.target_layer, nn.Conv2d):
                self.target_layer.weight.data = quantized_f.to(self.device)
            else: 
                print("Failure: updateWeight(self, newBIT)")
        
        else: 
            head = 0
            pos = 0

            for module in self.model.children():

                # White-Box: Update weights
                weight = module.weight.data

                # Recover 1d array into some layers of 1d array
                tail = self.list_sep[pos]
                layer_weight = new_BIT[head:tail].astype(np.float32)
                layer_weight = torch.from_numpy(layer_weight)

                # White-Box: Update weights
                quantized_f = binary_to_weight32(weight, self.qbits, layer_weight, self.dim_storage[pos])
                module.weight.data = quantized_f
                head = tail
                pos+=1

        return None

    
    def L1_BIT(self, adv_BIT):
        return np.not_equal(self.BIT_array,adv_BIT).astype(int).sum()


    #Model evaluation
    def check_accuracy(self):
        num_correct = 0
        num_sample = 0
        self.model.eval()  #only affecting Dropout and batch normalization layer

        with torch.no_grad():   #no need to cal grad
            for image,label in self.UNTARGETED_loader:
                image= image.to(self.device)
                label= label.to(self.device)
            
                # T x N x 2312 => N x 2312
                out_firing = self.model(image)
                #64x10 output
                _ , prediction = out_firing.max(1)  #64x1 (value in 2nd dimension)
                num_correct += (prediction==label).sum()
                num_sample += prediction.size(0)  #64 (value in 1st dimension)

        return (num_correct/num_sample).item()
    
    def fitness_fn(self,Pop:np): #np.type
        
        # Calculate fitness score for all 50 Pop samples
        Score = []
        for i in range(Pop.shape[0]):

            ####################################
            # L1 D(x',x) (raw BIT - adv BIT)
            l1_norm = self.L1_BIT(Pop[i])
            ####################################

            ####################################
            # Using new BIT_array for self.model
            # Testing with 128 random image
            self.updateWeight(Pop[i])
            Loss = self.check_accuracy()
            print(Loss)
            ####################################

            ####################################
            if (l1_norm <= self.epsil):
                Score +=[self.epsil*Loss]

            else:
                Score +=[(l1_norm + l1_norm*Loss)]
                #Score +=[l1_norm*Loss]   
            ####################################

        #print(Score)
        return np.array(Score) #np.type
    

    def selection(self, Pop:np):  #np.type
        
        # score on NEW population
        scores = self.fitness_fn(Pop)
        best_individual = min(scores)

        # re-indexing
        index = np.argsort(scores)
        Pop = Pop[index]
        
        # retain_len
        retain_len = int(self.retain_best*Pop.shape[0])
        #print(best_individual)
        
        return best_individual, list(Pop[:retain_len]) #List_type
        

    def crossover(self, parent1:np, parent2:np): #np.type
        
        # picking Dad,Mom from Male,Female Populations (All combinations: (Pop/2)^2)
        split_Pop = int(self.population_size*self.retain_best/2)
        dad = parent1[np.random.randint(0,split_Pop)]
        mom = parent2[np.random.randint(0,split_Pop)]

        """
        # K1-point Cross-over
        K1 = int(len(self.BIT_array)/2)
        child1, child2 = np.copy(mom), np.copy(dad)
        child1[:K1] = dad[:K1]
        child2[:K1] = mom[:K1]
        """

        """
        # K2-point Cross-over
        K1 = int(len(self.BIT_array)/3)
        K2 = int(len(self.BIT_array)/3)*2
        child1, child2 = np.copy(mom), np.copy(dad)
        child1[K1:K2] = dad[K1:K2]
        child2[K1:K2] = mom[K1:K2]
        """
        
        # Binomial Kn-point Cross-over
        select_mask = np.random.binomial(1, 0.5, size=len(self.BIT_array)).astype('bool')
        child1, child2 = np.copy(mom), np.copy(dad)
        child1[select_mask] = dad[select_mask]
        child2[select_mask] = mom[select_mask]
        
        
        return child1, child2
    

    def main(self):
        
        gen_evolution_score = []
        Curr_Pop = self.init_Pop()
        for i in range(self.n_generations):

            new_Pop = []
            best_guy, Pop = self.selection(Curr_Pop)
            new_Pop += Pop #List type
            gen_evolution_score += [best_guy]

            # Male/Female gene crossingover (Eugenics)
            split_Pop = int(self.population_size*self.retain_best/2)

            # Only best Male and Female can give birth
            males_parent = np.array(Pop[:split_Pop])  #np.type
            females_parent = np.array(Curr_Pop[split_Pop:])  #np.type

            # Random swap 5/15 elements between males and females (transgender: male too strong gene)
            # Generate 5 random position indices to swap
            indices = np.random.choice(int(self.population_size*self.retain_best/2), size=5, replace=False)
            males_parent[indices], females_parent[indices] = females_parent[indices], males_parent[indices]

            # Making 2 new Babies every time if community can support: (1-0.6)*50 = 20 (10 Loops)

            idx_1 = 1
            idx_2 = 2
            idx_3 = 3
            
            while (len(new_Pop) < self.population_size):
                child1, child2 = self.crossover(males_parent,females_parent)
                #child1 = self.reduced_mutation(child1)
                #child2 = self.reduced_mutation(child1)

                mut1 = self.reduced_mutation(np.array(new_Pop[idx_1]))
                mut2 = self.reduced_mutation(np.array(new_Pop[idx_2]))
                mut3 = self.reduced_mutation(np.array(new_Pop[idx_3]))

                new_Pop[idx_1] = list(mut1)
                new_Pop[idx_2] = list(mut2)
                new_Pop[idx_3] = list(mut3)

                idx_1 += 3
                idx_2 += 3
                idx_3 += 3

                new_Pop += [list(child1)]
                new_Pop += [list(child2)]

            # New generation
            Curr_Pop = np.array(new_Pop)
            print(f"Finish Gen: {i}")

        # Re-indexing on final population
        scores = self.fitness_fn(Curr_Pop)
        index = np.argsort(scores)
        Curr_Pop = Curr_Pop[index]

        # Return model, flipped bit, bit position
        self.updateWeight(Curr_Pop[0])
        
        return self.model, self.L1_BIT(Curr_Pop[0]), len(Curr_Pop[0]), np.array(gen_evolution_score)

def BITS_To_1D(model:nn.Module, layer_type:nn.Module, layer_idx:int, qbits:int, BITS_by_layer=False):

    # BITS for One Layer
    if (BITS_by_layer):
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

            return new_BIT, bit_shape, None, target_layer

        else:
            print("No layer found in the model.")

    # BITS for All Layer 
    else:  

        dim_storage = []
        BIT_array = []
        list_sep = []

        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                print(name)
                # Transform to array (3D matrix => 1D matrix)
                weight = module.weight.data
                weight1d, bit_shape = quantize_to_binary(weight, qbits)
                dim_storage += [bit_shape]

                # All layer weights matrix collapse to one 1d array
                BIT_array += weight1d.tolist()

                # TAIL positsion of curr layer (not HEAD of next layer)   
                list_sep += [len(BIT_array)]
                
        new_BIT = np.array(BIT_array).astype(np.float32)
        list_sep = np.array(list_sep).astype(np.int32)

        return new_BIT, dim_storage, list_sep, None


