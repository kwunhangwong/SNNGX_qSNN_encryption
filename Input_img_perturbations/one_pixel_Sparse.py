class GA_perturb_img():
    
    def __init__(self, x, y, model, encoder, T, epsil=5, gau_var=0.15 ,   #var 0.15  #mean -.02 #0.6  #epsilon = 20
                 n_generations=25, population_size=50, gau_mean=0, retain_best=0.6, mutate_chance=0.05):  #x= 
        
        
        # Initialization
        self.x = x
        self.y = y
        self.model = model
        self.T = T
        self.epsil = epsil
        
        # Population (must be even number)
        self.n_generations = n_generations
        self.population_size = population_size
        
        # Evolve
        self.retain_best = retain_best
        
        # Mutation
        self.mutate_chance = mutate_chance
        self.gau_var = gau_var
        self.gau_mean = gau_mean
        
    
    def init_Pop(self):
        
        adam_and_eve = []
        for i in range(int(self.population_size/2)):
            
            x_adv = np.zeros(784)
            x_adv = self.mutation(x_adv)
            adam_and_eve.append(x_adv)
            
            x_adv = self.x.clone().detach().cpu().numpy()
            x_adv = self.mutation(x_adv)
            adam_and_eve.append(x_adv)
        
        return np.array(adam_and_eve)
            
            
    def mutation(self, x_adv): #np.type
        
        # Each pixel has mutate_chance% to change
        s = np.random.normal(self.gau_mean, self.gau_var, len(x_adv))
        mutate_p = np.random.binomial(1, self.mutate_chance,len(x_adv))
        # Bounding Inject noises
        x_adv = np.clip(s*mutate_p+x_adv, 0, 1)
        
        return x_adv  #np.type
    
    def fitness_fn(self,Pop): #np.type
        
        origin = self.x.clone().detach().cpu().numpy()
        img = torch.from_numpy(Pop).float().to('cuda')
        
        out_firing = 0.
        for t in range(T):
            encoded_img = encoder(img)
            # Forward Network Batch_size = Pop.shape[0]
            out_firing += model(encoded_img)

        # Firing rate # Population_size x 10
        out_firing = out_firing / T
        functional.reset_net(model)
        out_firing = out_firing.detach().cpu().numpy()
        
        Score = []
        for i in range(Pop.shape[0]):
            # D(x',x) (raw x - x_adv)
            l2_norm = np.sqrt(np.square(origin-Pop[i]).sum())
            
            # M*L(x')  max(firing_correct - firing_2nd_largest, 0)
            Loss = max(out_firing[i][self.y]-np.sort(out_firing[i])[-2], 0)
            
            
            # sqrt(epsil) is defined with reference to sqrt(784)
            if (Loss == 0):
                M = 0
            else:
                M = self.epsil/Loss
            
            
            # Fitness = D(x',x) + M*L(x')
            Score +=[(l2_norm + M*Loss)]
        
        #print best_score
        #print(min(Score))
        
        return np.array(Score)  #np.type
    
    def selection(self, Pop):  #np.type
        
        # score on NEW population
        scores = self.fitness_fn(Pop)
        
        # re-indexing
        index = np.argsort(scores)
        Pop = Pop[index]
        
        # retain_len
        retain_len = int(self.retain_best*Pop.shape[0])
        
        return list(Pop[:retain_len]) #List_type
        
    #HARD-Code
    def crossover(self, parent1, parent2): #np.type
        

        """
        # 2-point crossover
        K1 = 261
        K2 = 523
        
        # 001100 and 110011
        gene1 = np.zeros(784)
        gene2 = np.ones(784)
        gene1[K1:K2] = 1
        gene2[K1:K2] = 0
        
        """

        # Parents
        dad = parent1[np.random.randint(0,len(parent1))]
        mom = parent2[np.random.randint(0,len(parent2))]
        
        """
        # Crossover 
        child1 = gene1*dad + gene2*mom
        child2 = gene2*dad + gene1*mom  
        """

        select_mask = np.random.binomial(1, 0.5, size=784).astype('bool')
        child1, child2 = np.copy(mom), np.copy(dad)
        child1[select_mask] = dad[select_mask]
        child2[select_mask] = mom[select_mask]
        
        
        return child1, child2
        
    
    
    def main(self):
        
        Curr_Pop = self.init_Pop()
        for i in range(self.n_generations):
            new_Pop = []
            new_Pop += self.selection(Curr_Pop)  #List type

            # Avoid self-self gene crossingover
            parent1 = Curr_Pop[:int(self.population_size/2)]  #np.type
            parent2 = Curr_Pop[int(self.population_size/2):]  #np.type
            
            #Now best 40% (20)
            #Remember to modify retain_elite to make multiple of 2, otherwie population will explode
            while (len(new_Pop) < self.population_size):
                child1, child2 = self.crossover(parent1,parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_Pop += [list(child1)]
                new_Pop += [list(child2)]
            
            #New generation
            Curr_Pop = np.array(new_Pop)
            
        return Curr_Pop[0]
        