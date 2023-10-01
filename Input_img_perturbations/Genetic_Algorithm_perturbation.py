import numpy as np

import torch
import torch.nn as nn

class GA_SparseAttack_L0: #Untargeted: Gen 25, eps = 5000, 100/10000
    
    def __init__(self, X, y , model:nn.Module,
                 epsil=1800, n_generations=25, population_size=100,      # epsil = L1/Hamming Distance bound
                 retain_best=0.4, mutate_chance=0.05):  

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
        self.L0_LARGE = np.max(self.BIT_array)
    
        
    def init_Pop(self):
        
        adam_and_eve = []
        for i in range(self.population_size):
            # Part Two (All flipped weight)
            BIT_array_adv = self.BIT_array.copy()
            counter =0
            while(counter<=10):
                BIT_array_adv = self.rectangular4D_mutation(BIT_array_adv)
                BIT_array_adv = self.LinearMovement4D_mutation(BIT_array_adv)
                counter += 1
            adam_and_eve.append(BIT_array_adv)
        return np.array(adam_and_eve)
    
    
    def rectangular4D_mutation(self,BIT_array_adv:np):
        
        # channel = np.random.randint(2)
        img_size = self.dim_storage[4]
        time_step = self.dim_storage[0]

        # 15,1,2,34,34 -> 15,2,34,34
        BIT_array_adv = np.squeeze(BIT_array_adv.reshape(self.dim_storage))
        step = np.random.randint(4)      #step 0-2
        direction = np.random.randint(2) #left or right
        height = np.random.randint(2,img_size//3)  #size 1-5
        width = np.random.randint(2,img_size//3)   #size 1-5
        start_index = np.random.randint(step*(height-1), 
                                        (img_size*img_size-(step*(height-1)+img_size*(height-1)+width))) 
        
        for i in range(time_step):
            # _2D_array = BIT_array_adv[i][channel].reshape(-1)
            # mutated = self.rectangular2D_mutation_edge(start_index,_2D_array,height,width,step,direction)

            pos_2D_array = BIT_array_adv[i][0].reshape(-1)
            neg_2D_array = BIT_array_adv[i][1].reshape(-1)

            mutated_pos = self.rectangular2D_mutation_edge(start_index,pos_2D_array,height,width,step,direction)
            mutated_neg = self.rectangular2D_mutation_edge(start_index,neg_2D_array,height,width,step,direction)

            BIT_array_adv[i][0] = mutated_pos.reshape(img_size,img_size)
            BIT_array_adv[i][1] = mutated_neg.reshape(img_size,img_size)

        return BIT_array_adv.reshape(-1)


    def rectangular2D_mutation_edge(self,start_index, BIT_array_adv:np,height:int,width:int,step,direction):
        img_size = self.dim_storage[4]
        if direction == 1: # "right edge"
            for i in range(height):
                starting_point = (start_index + img_size*i + step*i)
                ending_point = (starting_point + width)
                for j in range(starting_point,ending_point):
                    # BIT_array_adv[j] *= -1
                    BIT_array_adv[j] = np.sign(BIT_array_adv[j])*self.L0_LARGE*-1

        elif direction == 0: # "left edge"
            for i in range(height): 
                starting_point = ( start_index + img_size*i + step*(height-1-i) )
                ending_point = starting_point + width
                for j in range(starting_point,ending_point):
                    # BIT_array_adv[j] *= -1
                    BIT_array_adv[j] = np.sign(BIT_array_adv[j])*self.L0_LARGE*-1

        return BIT_array_adv
    
    def reduced_mutation(self, BIT_array_adv:np): #np.type

        # mask equal*1, not equal*-1*0.05(chance)
        mutate_mask = (self.BIT_array!=BIT_array_adv).astype(int)  
        flipping_position = np.random.binomial(1, self.mutate_chance,len(BIT_array_adv))
        places_flip = mutate_mask*flipping_position
        val, = np.where(places_flip == 1)

        BIT_array_adv[val] = self.BIT_array[val]   

        return BIT_array_adv #np.type

    def LinearMovement4D_mutation(self,BIT_array_adv:np):
        
        # channel = np.random.randint(2)
        img_size = self.dim_storage[4]
        time_step = self.dim_storage[0]

        # 15,1,2,34,34 -> 15,2,34,34
        BIT_array_adv = np.squeeze(BIT_array_adv.reshape(self.dim_storage))
        
        for i in range(time_step):

            pos_2D_array = BIT_array_adv[i][0].reshape(-1)
            neg_2D_array = BIT_array_adv[i][1].reshape(-1)

            # Get all current pixels
            pos_indices = np.where(pos_2D_array >= 1)
            neg_indices = np.where(neg_2D_array >= 1)

            # Remove the original image
            pos_2D_array[pos_indices] *= -1
            neg_2D_array[neg_indices] *= -1

            # NEw position val (POS)
            val, = pos_indices 

            move_vertical = np.random.randint(-img_size//4,img_size//4)  #size 1-5
            move_horizontal = np.random.randint(-img_size//4,img_size//4)   #size 1-5
            
            val = val + move_horizontal + move_vertical*img_size
            val = np.clip(val,0,len(pos_2D_array)-1)
            val = np.unique(val)

            # pos_2D_array[(val,)] *= -1 
            pos_2D_array[(val,)] = np.sign(pos_2D_array[(val,)])*self.L0_LARGE*-1

            # NEw position val (NEG)
            val, = neg_indices 

            move_vertical = np.random.randint(-img_size//5,img_size//5)  #size 1-5
            move_horizontal = np.random.randint(-img_size//5,img_size//5)   #size 1-5
            
            val = val + move_horizontal + move_vertical*img_size
            val = np.clip(val,0,len(neg_2D_array)-1)
            val = np.unique(val)
            
            # neg_2D_array[(val,)] *= -1
            pos_2D_array[(val,)] = np.sign(pos_2D_array[(val,)])*self.L0_LARGE*-1

            BIT_array_adv[i][0] = pos_2D_array.reshape(img_size,img_size)
            BIT_array_adv[i][1] = neg_2D_array.reshape(img_size,img_size)

        return BIT_array_adv.reshape(-1)
    
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
                adv_img = adv_img.clamp(min=0)
                out_firing = torch.squeeze(self.model(adv_img))   #1x10 

                Loss = out_firing[self.y].item()
                val , _ = torch.sort(out_firing)
                Loss = max(out_firing[self.y].item()-val[-2].item(), 0)

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

                mut1 = self.rectangular4D_mutation(np.array(new_Pop[idx_1]))
                mut2 = self.rectangular4D_mutation(np.array(new_Pop[idx_2]))
                mut3 = self.rectangular4D_mutation(np.array(new_Pop[idx_3]))

                mut1 = self.LinearMovement4D_mutation(np.array(new_Pop[idx_1]))
                mut2 = self.LinearMovement4D_mutation(np.array(new_Pop[idx_2]))
                mut3 = self.LinearMovement4D_mutation(np.array(new_Pop[idx_3]))

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
    
    def Edges_only(self): #np.type
        adv_img = self.BIT_array.copy()
        counter =0
        while(counter <=20):
            adv_img = self.rectangular4D_mutation(adv_img)
            counter+=1
        adv_img = (adv_img + 1) / 2
        adv_img = torch.from_numpy(adv_img.astype(np.float32).reshape(self.dim_storage)).to(self.device)
        adv_img = adv_img.clamp(min=0)
        return adv_img
    
    def Move_only(self): #np.type
        adv_img = self.BIT_array.copy()
        counter =0
        while(counter <=5):
            adv_img = self.LinearMovement4D_mutation(adv_img)
            counter+=1
        adv_img = (adv_img + 1) / 2
        adv_img = torch.from_numpy(adv_img.astype(np.float32).reshape(self.dim_storage)).to(self.device)
        adv_img = adv_img.clamp(min=0)
        return adv_img


class GA_SparseAttack_L0_targeted: 
    
    def __init__(self, X, label, target , model:nn.Module,
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
        self.label = label.item()

        # Adversarial Image (overall benchmark) ({1,0} -> {+1,-1})
        self.BIT_array = 2*(X.to(torch.device('cpu')).numpy().astype(np.float32).reshape(-1))- 1
        self.dim_storage = X.shape  # 15,1,2,34,34

    def init_Pop(self):
        
        adam_and_eve = []
        for i in range(self.population_size):
            # Part Two (All flipped weight)
            BIT_array_adv = self.BIT_array.copy()
            counter =0
            while(self.check_accuracy(BIT_array_adv)==1 and counter <=20):
                BIT_array_adv = self.rectangular4D_mutation(BIT_array_adv)
                # BIT_array_adv = self.permute_mutation(BIT_array_adv)
                counter+=1
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

    def rectangular4D_mutation(self,BIT_array_adv:np):
        
        # channel = np.random.randint(2)
        img_size = self.dim_storage[4]
        time_step = self.dim_storage[0]
        # 15,1,2,34,34 -> 15,2,34,34
        BIT_array_adv = np.squeeze(BIT_array_adv.reshape(self.dim_storage))
        step = np.random.randint(4)      #step 0-2
        direction = np.random.randint(2) #left or right
        height = np.random.randint(2,img_size//5)  #size 1-5
        width = np.random.randint(2,img_size//5)   #size 1-5
        start_index = np.random.randint(step*(height-1), 
                                        (img_size*img_size-(step*(height-1)+img_size*(height-1)+width))) 
        
        for i in range(time_step):
            # _2D_array = BIT_array_adv[i][channel].reshape(-1)
            # mutated = self.rectangular2D_mutation_edge(start_index,_2D_array,height,width,step,direction)

            pos_2D_array = BIT_array_adv[i][0].reshape(-1)
            neg_2D_array = BIT_array_adv[i][1].reshape(-1)

            # step = np.random.randint(4)      #step 0-2
            # direction = np.random.randint(2) #left or right
            # height = np.random.randint(2,img_size//5)  #size 1-5
            # width = np.random.randint(2,img_size//5)   #size 1-5
            # start_index = np.random.randint(step*(height-1), 
            #                     (img_size*img_size-(step*(height-1)+img_size*(height-1)+width))) 

            mutated_pos = self.rectangular2D_mutation_edge(start_index,pos_2D_array,height,width,step,direction)
            mutated_neg = self.rectangular2D_mutation_edge(start_index,neg_2D_array,height,width,step,direction)

            BIT_array_adv[i][0] = mutated_pos.reshape(img_size,img_size)
            BIT_array_adv[i][1] = mutated_neg.reshape(img_size,img_size)

        return BIT_array_adv.reshape(-1)


    def rectangular2D_mutation_edge(self,start_index, BIT_array_adv:np,height:int,width:int,step,direction):
        img_size = self.dim_storage[4]
        if direction == 1: # "right edge"
            for i in range(height):
                starting_point = (start_index + img_size*i + step*i)
                ending_point = (starting_point + width)
                for j in range(starting_point,ending_point):
                    BIT_array_adv[j] *= -1

        elif direction == 0: # "left edge"
            for i in range(height): 
                starting_point = ( start_index + img_size*i + step*(height-1-i) )
                ending_point = starting_point + width
                for j in range(starting_point,ending_point):
                    BIT_array_adv[j] *= -1

        return BIT_array_adv


    def LinearMovement4D_mutation(self,BIT_array_adv:np):
        
        # channel = np.random.randint(2)
        img_size = self.dim_storage[4]
        time_step = self.dim_storage[0]

        # 15,1,2,34,34 -> 15,2,34,34
        BIT_array_adv = np.squeeze(BIT_array_adv.reshape(self.dim_storage))
        
        for i in range(time_step):

            pos_2D_array = BIT_array_adv[i][0].reshape(-1)
            neg_2D_array = BIT_array_adv[i][1].reshape(-1)

            # Get all current pixels
            pos_indices = np.where(pos_2D_array >= 1)
            neg_indices = np.where(neg_2D_array >= 1)

            # Remove the original image
            pos_2D_array[pos_indices] *= -1
            neg_2D_array[neg_indices] *= -1

            # NEw position val (POS)
            val, = pos_indices 

            move_vertical = np.random.randint(-img_size//5,img_size//5)  #size 1-5
            move_horizontal = np.random.randint(-img_size//5,img_size//5)   #size 1-5
            
            val = val + move_horizontal + move_vertical*img_size
            val = np.clip(val,0,len(pos_2D_array)-1)
            val = np.unique(val)

            pos_2D_array[(val,)] *= -1

            # NEw position val (NEG)
            val, = neg_indices 

            move_vertical = np.random.randint(-img_size//5,img_size//5)  #size 1-5
            move_horizontal = np.random.randint(-img_size//5,img_size//5)   #size 1-5
            
            val = val + move_horizontal + move_vertical*img_size
            val = np.clip(val,0,len(neg_2D_array)-1)
            val = np.unique(val)
            
            neg_2D_array[(val,)] *= -1

            BIT_array_adv[i][0] = pos_2D_array.reshape(img_size,img_size)
            BIT_array_adv[i][1] = neg_2D_array.reshape(img_size,img_size)

        return BIT_array_adv.reshape(-1)

    def L1_BIT(self, adv_BIT):
        return np.not_equal(self.BIT_array,adv_BIT).astype(int).sum()


    def check_accuracy(self,adv_BIT):
        with torch.no_grad():   
            adv_img = (adv_BIT + 1) / 2
            adv_img = torch.from_numpy(adv_img.astype(np.float32).reshape(self.dim_storage)).to(self.device)
            out_firing = torch.squeeze(self.model(adv_img))   #1x10 

            Loss_tar = 1 - out_firing[self.y].item()
            print(Loss_tar)

            return Loss_tar
            # Loss_label = out_firing[self.label].item()
            # Loss = (Loss_label+Loss_tar)/2
            # return Loss

    def fitness_fn(self,Pop:np): #np.type
        
        # Calculate fitness score for all 50 Pop samples
        Score = []
        for i in range(Pop.shape[0]):

            ####################################
            # L1 D(x',x) (raw BIT - adv BIT)
            l1_norm = self.L1_BIT(Pop[i])
            ####################################
            # rmb the image is now {+1,-1}, not {1,0}
            Loss = self.check_accuracy(Pop[i])

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
            indices = np.random.choice(int(self.population_size*self.retain_best/2), size=int(self.population_size*self.retain_best/6), replace=False)
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
                
                mut1 = self.rectangular4D_mutation(np.array(new_Pop[idx_1]))
                mut2 = self.rectangular4D_mutation(np.array(new_Pop[idx_2]))
                mut3 = self.rectangular4D_mutation(np.array(new_Pop[idx_3]))

                mut1 = self.LinearMovement4D_mutation(np.array(new_Pop[idx_1]))
                mut2 = self.LinearMovement4D_mutation(np.array(new_Pop[idx_2]))
                mut3 = self.LinearMovement4D_mutation(np.array(new_Pop[idx_3]))
                

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

