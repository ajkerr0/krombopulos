"""


"""

import numpy as np
import matplotlib.pyplot as plt

import ballnspring

class Evolution:
    """A genetic algorithm class, using binary strings as individuals.
    
    Arguments:
        isize (int): Size of the individual strings.
        psize (int): Size of the population.
        fitfunc (function): Fitness function; returns high values for high fitness.
    
    Keywords:
        abet (int): Size of the alphabet i.e. how many possible integers in a chromosome.
        nepoch (int): Number of epochs in the calculation.
        nelite (int): Number of chromosomes preserved from the previous generation during elitism.
        tour (bool): When True, apply tournament rules."""
    
    def __init__(self, isize, psize, fitfunc, abet=2, nepoch=50, nelite=3, tour=True):
        self.fitfunc = fitfunc
        self.abet = abet
        self.nepoch = nepoch
        self.nelite = nelite
        self.tour = tour
        if psize%2 == 1:
            psize += 1
        self.pop = np.random.randint(0,abet,(psize,isize))
        self.mrate = 1/isize
#        print(self.pop)
        
    @staticmethod
    def select_parents(pop, fitness):
        """Return the parents of the next generation
        using fitness proportional selection."""
        
        #scale the fitnesses such that the highest value is the population size
            #this guarantees there will be enough random samples
        #ignore individuals with new fitness < 1 as parents for new generation
        #add number of copies of the individuals based on their new fitness to be randomly selected
        
        fitness = fitness/np.sum(fitness)
        fitness = pop.shape[0]*fitness/fitness.max()
        
        newpop = []
        for i in range(pop.shape[0]):
            if np.round(fitness[i]) >= 1:
                newpop.extend(np.kron(np.ones((np.round(fitness[i]),1)), pop[i,:]))
        
        newpop = np.asarray(newpop)     
        newpop = newpop.astype(int)
        
        indices = np.arange(newpop.shape[0])
        np.random.shuffle(indices)
        
        return newpop[indices[:pop.shape[0]]]
    
    @staticmethod
    def crossover(pop):
        """Return offspring of input population by
        performing single point crossover."""
        
        newpop = np.zeros(pop.shape, dtype=int)
        cross_point = np.random.randint(0, pop.shape[1], pop.shape[0])
        
        for i in range(0, pop.shape[0], 2):
            newpop[i  , :cross_point[i]] = pop[i  , :cross_point[i]]
            newpop[i  , cross_point[i]:] = pop[i+1, cross_point[i]:]
            newpop[i+1, :cross_point[i]] = pop[i+1, :cross_point[i]]
            newpop[i+1, cross_point[i]:] = pop[i  , cross_point[i]:]
        
        return newpop
        
    def mutate(self, pop):
        """Return mutated population"""
        
        whereMu = np.random.rand(pop.shape[0], pop.shape[1])
#        pop[np.where(whereMu < self.mrate)] = 1 - pop[np.where(whereMu < self.mrate)]
        pop[np.where(whereMu < self.mrate)] = np.random.randint(0, self.abet, pop[np.where(whereMu < self.mrate)].shape)
        return pop
     
    def elitism(self, newpop, fitness):
        """Return the population with random individuals replaced by the nelite
        individuals from the previous generation."""
        
        best = np.argsort(fitness)
        best = self.pop[best[-self.nelite:]]
        indices = np.arange(newpop.shape[0])
        np.random.shuffle(indices)
        newpop = newpop[indices]
        newpop[:self.nelite] = best
        return newpop
        
    def tournament(self, newpop, fitness):
        """Return the results of a tournament between the new generation and
        its preceeding one."""
        return newpop
        
        
    def evolve(self):
        """Run the genetic algorithm"""
        
        bestfit = np.zeros(self.nepoch)
        
        for i in range(self.nepoch):
            
            #get the fitness of the current population
            fitness = self.fitfunc(self.pop)
            
            #select the parents of the next generation
            newpop = self.select_parents(self.pop, fitness)
            
            #perform crossover, mutation
            newpop = self.crossover(newpop)
            newpop = self.mutate(newpop)
            
            #apply elitism and host tournaments
            if self.nelite > 0:
                newpop = self.elitism(newpop, fitness)
            if self.tour:
                newpop = self.tournament(newpop, fitness)
                
            self.pop = newpop
            bestfit[i] = np.max(fitness)
                    
        print(self.pop)
        print(self.fitfunc(self.pop))
            
        plt.plot(np.arange(1,self.nepoch+1), bestfit, '-rx', linewidth=4., markersize=15.)
        plt.xlabel("Generation count", fontsize=15.)
        plt.ylabel("Max fitness in population", fontsize=15.)
        plt.suptitle("Max fitness vs. Generation", fontsize=18.)
        plt.show()
        
class Evo2(Evolution):
    """A GA class that inherits Evolution, but individuals are effectively comprised of
    2 chromosomes
    
    Arguments:
        isize (int): Size of the individual strings.
        psize (int): Size of the population.
        fitfunc (function): Fitness function; returns high values for high fitness.
        csplit (int): Index of the beginning of the 2nd chromosome in the individual strings.
    
    Keywords:
        abet (int): Size of the alphabet i.e. how many possible integers in a chromosome.
        nepoch (int): Number of epochs in the calculation.
        nelite (int): Number of chromosomes preserved from the previous generation during elitism.
        tour (bool): When True, apply tournament rules."""
        
    def __init__(self, isize, psize, fitfunc, csplit, abet=2, nepoch=50, nelite=3, tour=True):
        super().__init__(isize, psize, fitfunc, abet=abet,
                           nepoch=nepoch, nelite=nelite, tour=tour)
        self.csplit = csplit
      
    def crossover(self, pop):
        """Perform multipoint crossover on both halves of the individuals"""
        
        newpop = np.zeros(pop.shape, dtype=int)
        
#        cross_point = np.random.randint(0, pop.shape[1]//2, (pop.shape[0],2))
#        cross_point[:,1] += pop.shape[1]//2
        cross_point = np.zeros((pop.shape[0],2), dtype=int)
        cross_point[:,0] = np.random.randint(0, self.csplit, (pop.shape[0]))
        cross_point[:,1] = np.random.randint(self.csplit, pop.shape[1], (pop.shape[0]))
        
        for i in range(0, pop.shape[0], 2):
            p1, p2 = cross_point[i]
            newpop[i  ,   :p1] = pop[i  ,   :p1]
            newpop[i  ,   p2:] = pop[i  ,   p2:]
            newpop[i  , p1:p2] = pop[i+1, p1:p2]
            newpop[i+1,   :p1] = pop[i+1,   :p1]
            newpop[i+1,   p2:] = pop[i+1,   p2:]
            newpop[i+1, p1:p2] = pop[i  , p1:p2]
            
        return newpop
        
def fourpeaks(pop, T=.15):
    
    T = np.ceil(pop.shape[0]*T)
    
    fitness = np.zeros(pop.shape[0])
    
    for i in range(pop.shape[0]):
        zeros = np.where(pop[i,:]==0)[0]
        ones =  np.where(pop[i,:]==1)[0]
        
        if ones.size > 0:
            consec0 = ones[0]
        else:
            consec0 = 0
            
        if zeros.size > 0:
            consec1 = pop.shape[1] - zeros[-1] - 1
        else:
            consec1 = 0
            
        if consec0 > T and consec1 > T:
            fitness[i] = np.maximum(consec0, consec1)+100
        else:
            fitness[i] = np.maximum(consec0, consec1)
            
    return fitness
    
def transmission_mass(pop, center=5, centermass=1.5, stiffval=20.):
    """Return the thermal conductivities of the chains with 
    with individuals as the leading strands."""
    
    slen = pop.shape[1]
    
    #determine masses and spring constants
    #start with center
    stiff = [[i,i+1] for i in range(center-1)]
    loose = [[i-1,i] for i in range(center, slen + center)]
    loose.append([0, center+slen])
    loose.extend([[i,i+1] for i in range(slen + center, 2*slen + center - 1)])
    
    size = center + 2*slen
    k = np.zeros((size,size))
    looseval = 5.
    for i,j in stiff:
        k[i,i] += stiffval
        k[j,j] += stiffval
        k[i,j] = -stiffval
        k[j,i] = -stiffval
    for i,j in loose:
        k[i,i] += looseval
        k[j,j] += looseval
        k[i,j] = -looseval
        k[j,i] = -looseval
        
    drivers = [[slen + center - 1],[2*slen + center - 1]]
    crossings = [[center+slen,0],[center-1, center]]
    
    centermass = [centermass]*center
    
    fitness = np.zeros(pop.shape[0])
    
    for i in range(pop.shape[0]):
        
        mass = centermass + list(pop[i] + 1) + list(pop[i][::-1] + 1)
        
        fitness[i] = abs(ballnspring.kappa(mass, k, drivers, crossings, gamma=1.))
        
    return fitness
    
def spring_constant(segments):
    """Return the spring constant as a function of the segment of the chromotid"""
    return segments + 1.
    
def transmission_mass_and_spring_even(pop, center=5, centermass=1.5, stiffval=20.):
    """Return the fitness of the population of individuals with variable
    mass AND spring strength."""
    
    #check for valid input
    if pop.shape[1]%2 != 0:
        raise ValueError("Invalid input to fitness function, should be same number of variable springs as \
                          there are masses")
                          
    slen = pop.shape[1]//2
    
    #determine masses and spring constants from chromosomes
    #start with center
    stiff = [[i,i+1] for i in range(center-1)]
    loose = [[i-1,i] for i in range(center, slen + center)]
    loose.append([0, center+slen])
    loose.extend([[i,i+1] for i in range(slen + center, 2*slen + center - 1)])
    loosevals = spring_constant(np.tile(pop[:,slen:], (1,2)))
    
    size = center + 2*slen
        
    drivers = [[slen + center - 1],[2*slen + center - 1]]
    crossings = [[center+slen,0],[center-1, center]]
#    crossings = [[0,center+slen],[center, center-1]]
    
    centermass = [centermass]*center
    
    fitness = np.zeros(pop.shape[0])
    
    for m in range(pop.shape[0]):
        
        mass = centermass + list(pop[m,:slen] + 1) + list(pop[m,:slen][::-1] + 1)
        
        k = np.zeros((size,size))
        for i,j in stiff:
            k[i,i] += stiffval
            k[j,j] += stiffval
            k[i,j] = -stiffval
            k[j,i] = -stiffval
            
        for lindices, lval in zip(loose, loosevals[m]):
            i,j = lindices
            k[i,i] += lval
            k[j,j] += lval
            k[i,j] = -lval
            k[j,i] = -lval

        fitness[m] = abs(ballnspring.kappa(mass, k, drivers, crossings, gamma=1.))
        
    return fitness
    
def transmission_mass_and_spring_odd(pop, center=5, centermass=1.5, endmass=.1, stiffval=20.):
    """Return the fitness of the population of individuals with variable
    mass AND spring strength."""
    
    #check for valid input
    if pop.shape[1]%2 == 0:
        raise ValueError("Invalid input to fitness function, should be one more spring than there \
                          variable masses.")
                          
    slen = (pop.shape[1]-1)//2
    
    #determine masses and spring constants from chromosomes
    #start with center
    stiff = [[i,i+1] for i in range(center-1)]
    loose = [[i-1,i] for i in range(center, slen + center)]
    loose.append([slen + center-1, 2*slen + center])
    loose.append([0, center+slen])
    loose.extend([[i,i+1] for i in range(slen + center, 2*slen + center - 1)])
    loose.append([2*slen+center-1, 2*slen + center+1])
#    loose.extend([[slen + center-1, 2*slen + center],[2*slen+center-1, 2*slen + center+1]])
    loosevals = spring_constant(np.tile(pop[:,slen:], (1,2)))
    
    size = center + 2*slen + 2
        
    drivers = [[slen + center - 1],[2*slen + center - 1]]
    crossings = [[center+slen,0],[center-1, center]]
#    crossings = [[0,center+slen],[center, center-1]]
    
#    print(stiff)
#    print(loose)
#    print(loosevals)
    
    centermass = [centermass]*center
    
    fitness = np.zeros(pop.shape[0])
    
    for m in range(pop.shape[0]):
        
        mass = centermass + list(pop[m,:slen] + 1.) + list(pop[m,:slen][::-1] + 1.) + [endmass]*2
        
        k = np.zeros((size,size))
        for i,j in stiff:
            k[i,i] += stiffval
            k[j,j] += stiffval
            k[i,j] = -stiffval
            k[j,i] = -stiffval
            
        for lindices, lval in zip(loose, loosevals[m]):
            i,j = lindices
            k[i,i] += lval
            k[j,j] += lval
            k[i,j] = -lval
            k[j,i] = -lval
            
#        print(mass)
#        print(k)

        fitness[m] = abs(ballnspring.kappa(mass, k, drivers, crossings, gamma=1.))
        
    return fitness
    
isize = 11
psize = 20

#a = Evolution(4, 20, transmission_mass, nepoch=1, abet=2)
#a = Evo2(isize, psize, transmission_mass_and_spring_even, isize//2, nepoch=10, abet=20)
a = Evo2(isize, psize, transmission_mass_and_spring_odd, (isize-1)//2, nepoch=100, abet=10)
a.evolve()

best = np.array([0, 1, 1, 1, 0, 9, 9, 7, 7, 5, 8,])

def bar_plot(fittest, title, x, y):
    """Plot a bar graph showing the distributions of the inputs."""
    
    fittest = fittest + 1.
    ind = np.arange(fittest.shape[0])
    width = .35
    
    fig,ax = plt.subplots()
    rects = ax.bar(ind + width, fittest, width, color='r')
    
    ax.set_ylabel(y, fontsize=16)
    ax.set_xlabel(x, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.set_xticks(ind + width*1.5)
    ax.set_yticks(np.arange(np.max(fittest)+2)+1)
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6'))
    
    plt.show()
    
bar_plot(best[:5], "Mass distribution of fittest side chain", "Mass Number", "Mass Value (mass units)")
bar_plot(best[5:], "Spring distribution of fittest side chain", "Spring Number", "Spring constant (spring units)")