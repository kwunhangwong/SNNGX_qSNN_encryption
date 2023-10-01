import numpy as np

import torch
import torch.nn as nn

class GA_SparseAttack_L0: #Untargeted: Gen 25, eps = 5000, 100/10000
    
    def __init__(self, X, y , model:nn.Module,
                 epsil=5000, n_generations=25, population_size=100,      # epsil = L1/Hamming Distance bound
                 retain_best=0.6, mutate_chance=0.05):  

        # Initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval() 

        # Population (must be even number)
        self.n_generations = n_generations
        self.population_size = population_size

        # Evolve
        self.retain_best = retain_best
        # Mutation
        self.mutate_chance = mutate_chance

        # Target Perturbation
        self.epsil = epsil

        # Adversarial Image (for fitness cal)
        self.X = X
        self.y = y.item()

        # Adversarial Image (overall benchmark) ({1,0} -> {+1,-1})
        self.BIT_array = 2*(X.to(torch.device('cpu')).numpy().astype(np.float32).reshape(-1))- 1
        self.dim_storage = X.shape
    

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

    
    def L1_BIT(self, adv_BIT):
        return np.not_equal(self.BIT_array,adv_BIT).astype(int).sum()


    def fitness_fn(self,Pop:np): #np.type
        
        # Calculate fitness score for all 50 Pop samples
        Score = []
        for i in range(Pop.shape[0]):

            ####################################
            # L1 D(x',x) (raw BIT - adv BIT)
            l1_norm = self.L1_BIT(Pop[i])
            ####################################
            # rmb the image is now {+1,-1}, not {1,0}
            with torch.no_grad():   

                adv_img = ((Pop[i]) + 1) / 2
                adv_img = torch.from_numpy(adv_img.astype(np.float32).reshape(self.dim_storage)).to(self.device)
                out_firing = torch.squeeze(self.model(adv_img))   #1x10 

                Loss = out_firing[self.y].item()
                print(Loss)

                if (l1_norm <= self.epsil):
                    Score +=[self.epsil*Loss]

                else:
                    Score +=[(l1_norm + l1_norm*Loss)]

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
        
        return best_individual, list(Pop[:retain_len]) #List_type
        

    def crossover(self, parent1:np, parent2:np): #np.type
        
        # picking Dad,Mom from Male,Female Populations (All combinations: (Pop/2)^2)
        split_Pop = int(self.population_size*self.retain_best/2)
        dad = parent1[np.random.randint(0,split_Pop)]
        mom = parent2[np.random.randint(0,split_Pop)]
        
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
        adv_img = (Curr_Pop[0] + 1) / 2
        adv_img = torch.from_numpy(adv_img.astype(np.float32).reshape(self.dim_storage)).to(self.device)
        adv_img = adv_img.clamp(min=0)
        
        return adv_img, self.L1_BIT(Curr_Pop[0]), len(Curr_Pop[0]), np.array(gen_evolution_score)



class GA_SparseAttack_L0_targeted: 
    
    def __init__(self, X, target , model:nn.Module,
                 epsil=5000, n_generations=25, population_size=100,      # epsil = L1/Hamming Distance bound
                 retain_best=0.6, mutate_chance=0.05):  

        # Initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval() 

        # Population (must be even number)
        self.n_generations = n_generations
        self.population_size = population_size

        # Evolve
        self.retain_best = retain_best
        # Mutation
        self.mutate_chance = mutate_chance

        # Target Perturbation
        self.epsil = epsil

        # Adversarial Image (for fitness cal)
        self.X = X
        self.y = target

        # Adversarial Image (overall benchmark) ({1,0} -> {+1,-1})
        self.BIT_array = 2*(X.to(torch.device('cpu')).numpy().astype(np.float32).reshape(-1))- 1
        self.dim_storage = X.shape
    

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

    
    def L1_BIT(self, adv_BIT):
        return np.not_equal(self.BIT_array,adv_BIT).astype(int).sum()


    def fitness_fn(self,Pop:np): #np.type
        
        # Calculate fitness score for all 50 Pop samples
        Score = []
        for i in range(Pop.shape[0]):

            ####################################
            # L1 D(x',x) (raw BIT - adv BIT)
            l1_norm = self.L1_BIT(Pop[i])
            ####################################
            # rmb the image is now {+1,-1}, not {1,0}
            with torch.no_grad():   

                adv_img = ((Pop[i]) + 1) / 2
                adv_img = torch.from_numpy(adv_img.astype(np.float32).reshape(self.dim_storage)).to(self.device)
                out_firing = torch.squeeze(self.model(adv_img))   #1x10 

                Loss = 1 - out_firing[self.y].item()
                print(Loss)

                if (l1_norm <= self.epsil):
                    Score +=[self.epsil*Loss]

                else:
                    Score +=[(l1_norm + l1_norm*Loss)]

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
        
        return best_individual, list(Pop[:retain_len]) #List_type
        

    def crossover(self, parent1:np, parent2:np): #np.type
        
        # picking Dad,Mom from Male,Female Populations (All combinations: (Pop/2)^2)
        split_Pop = int(self.population_size*self.retain_best/2)
        dad = parent1[np.random.randint(0,split_Pop)]
        mom = parent2[np.random.randint(0,split_Pop)]
        
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
        adv_img = (Curr_Pop[0] + 1) / 2
        adv_img = torch.from_numpy(adv_img.astype(np.float32).reshape(self.dim_storage)).to(self.device)
        # adv_img = adv_img.clamp(min=0)
        
        return adv_img, self.L1_BIT(Curr_Pop[0]), len(Curr_Pop[0]), np.array(gen_evolution_score)




