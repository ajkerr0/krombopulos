
import numpy as np
import matplotlib.pyplot as plt

class GA1:
    """A genetic algorithm with one chromosome type.
    
    Parameters:
    
        isize (int):
            Size of each of the individuals chromosomes.
        psize (int): 
            Size of the population.
        fitfunc (function): 
            Fitness function that returns high values for high fitness.  Must be callable with
              sequences of length isize as the parameter.
    
    Keywords:
    
        ndigit (int):
            Number of digits accessible in a chromosome.  Defaults to 2 for binary strings.
        nepoch (int): 
            Number of epochs in the algorithm.  Defaults to 100.
        nelite (int): 
            Number of chromosomes preserved from the previous generation during elitism.
            Defaults to 2.
        mrate (float):
            Fraction of n-bits that mutate, on average.  Defaults to 'None" which means the mutation
              rate flips one n-bit per chromosome on average."""
    
    def __init__(self, isize, psize, fitfunc, ndigit=2, nepoch=100, nelite=2, mrate=None):
        self.isize = isize
        self.fitfunc = fitfunc
        self.nepoch = nepoch
        self.nelite = nelite
        self.ndigit = ndigit
        if psize%2 == 1:
            psize += 1
        self.pop = np.random.randint(0,ndigit,(psize,isize))
        if mrate is None:
            self.mrate = 1/isize
        else:
            self.mrate = mrate
        
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
        """Return mutated population.  There is a chance to mutate back
        to the original digit."""
        
        whereMu = np.random.rand(pop.shape[0], pop.shape[1])
        muPop = np.where(whereMu < self.mrate)
        pop[muPop] = np.random.randint(0, self.ndigit, pop[muPop].shape)
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
                
            self.pop = newpop
            bestfit[i] = np.max(fitness)
                    
        print(self.pop)
        final_fitness = self.fitfunc(self.pop)
        print(final_fitness)
        best_index = np.argmax(final_fitness)
            
        self.best_index = best_index
        self.bestfit = bestfit
        self.best = self.pop[best_index]
        
    def plot(self, plotval, title=None, xlabel=None, ylabel=None, popfunc=None, savefile=None):
    
        if plotval == 0:
            
            if title is None:
                title = "Maximum fitness vs. Generation count"
            if xlabel is None:
                xlabel = "Generation count"
            if ylabel is None:
                ylabel = "Max fitness in population"
            
            fig = plt.figure()
            plt.plot(np.arange(1,self.bestfit.shape[0] + 1), self.bestfit, '-rx', linewidth=4., markersize=15.)
            plt.xlabel(xlabel, fontsize=15.)
            plt.ylabel(ylabel, fontsize=15.)
            plt.suptitle(title, fontsize=18.)
            plt.show()
            
        elif plotval == 1:
            
            if popfunc is None:
                fittest = self.pop[self.best_index]
            else:
                fittest = popfunc(self.pop[self.best_index])
                
            ind = np.arange(fittest.shape[0])
            width = .35
            fig,ax = plt.subplots()
            ax.bar(ind + width, fittest, width, color='r')
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_title(title, fontsize=20)
            ax.set_xticks(ind + width*1.5)
            ax.set_yticks(np.arange(np.max(fittest)+2)+1)
            ax.set_xticklabels(np.arange(1,fittest.shape[0]+1))
        
        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight')
        
class GA2:
    """A genetic algorithm with two chromosome types.
    
    Parameters:
    
        isize1 (int):
            Size of the first chromosome type.
        isize2 (int):
            Size of the second chromosome type.
        psize (int): 
            Size of the population.
        fitfunc (function): 
            Fitness function that returns high values for high fitness.  Must be callable with
              sequences of length isize1 and isize2 as the parameters.
    
    Keywords:
    
        ndigit1 (int):
            Number of digits accessible chromosome type 1.  Defaults to 2 for binary strings.
        ndigit2 (int):
            Number of digits accessible chromosome type 2.
        nepoch (int): 
            Number of epochs in the algorithm.  Defaults to 100.
        nelite (int): 
            Number of chromosomes preserved from the previous generation during elitism.
            Defaults to 2.
        mrate (float):
            Fraction of n-bits that mutate, on average.  Defaults to 'None" which means the mutation
              rate flips one n-bit per chromosome on average."""
    
    def __init__(self, isize1, isize2, psize, fitfunc, 
                 ndigit1=2, ndigit2=2, nepoch=100, nelite=2, mrate=None):
        self.isize1 = isize1
        self.isize2 = isize2
        self.psize = psize
        self.fitfunc = fitfunc
        self.nepoch = nepoch
        self.nelite = nelite
        self.ndigit1 = ndigit1
        self.ndigit2 = ndigit2
        if psize%2 == 1:
            psize += 1
        self.pop1 = np.random.randint(0,ndigit1,(psize,isize1))
        self.pop2 = np.random.randint(0,ndigit2,(psize,isize2))
        if mrate is None:
            self.mrate = 1/(isize1 + isize2)
        else:
            self.mrate = mrate
        
    def select_parents(self, fitness):
        """Return the parents of the next generation
        using fitness proportional selection."""
        
        #scale the fitnesses such that the highest value is the population size
            #this guarantees there will be enough random samples
        #ignore individuals with new fitness < 1 as parents for new generation
        #add number of copies of the individuals based on their new fitness to be randomly selected
        
        fitness = fitness/np.sum(fitness)
        fitness = self.psize*fitness/fitness.max()
        
        newpop1 = []
        newpop2 = []
        for i in range(self.psize):
            if np.round(fitness[i]) >= 1:
                newpop1.extend(np.kron(np.ones((np.round(fitness[i]),1)), self.pop1[i,:]))
                newpop2.extend(np.kron(np.ones((np.round(fitness[i]),1)), self.pop2[i,:]))
        
        newpop1, newpop2 = np.asarray(newpop1), np.asarray(newpop2) 
        newpop1, newpop2 = newpop1.astype(int), newpop2.astype(int)
        
        indices = np.arange(self.psize)
        np.random.shuffle(indices)
        
        return newpop1[indices[:self.psize]], newpop2[indices[:self.psize]]
    
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
        
    def mutate(self, pop1, pop2):
        """Return mutated population."""
        
        whereMu1 = np.random.rand(self.psize, self.isize1)
        whereMu2 = np.random.rand(self.psize, self.isize2)
        muPop1, muPop2 = np.where(whereMu1 < self.mrate), np.where(whereMu2 < self.mrate)
        pop1[muPop1] = np.random.randint(0, self.ndigit1, pop1[muPop1].shape)
        pop2[muPop2] = np.random.randint(0, self.ndigit2, pop2[muPop2].shape)
        return pop1, pop2
     
    def elitism(self, newpop1, newpop2, fitness):
        """Return the population with random individuals replaced by the nelite
        individuals from the previous generation."""
        
        best = np.argsort(fitness)
        best1, best2 = self.pop1[best[-self.nelite:]], self.pop2[best[-self.nelite:]]
        indices = np.arange(self.psize)
        np.random.shuffle(indices)
        newpop1, newpop2 = newpop1[indices], newpop2[indices]
        newpop1[:self.nelite], newpop2[:self.nelite] = best1, best2
        return newpop1, newpop2        
        
    def evolve(self):
        """Run the genetic algorithm"""
        
        bestfit = np.zeros(self.nepoch)
        
        for i in range(self.nepoch):
            
            #get the fitness of the current population
            fitness = self.fitfunc(self.pop1, self.pop2)
            
            #select the parents of the next generation
            newpop1, newpop2 = self.select_parents(fitness)
            
            #perform crossover, mutation
            newpop1, newpop2 = self.crossover(newpop1), self.crossover(newpop2)
            newpop1, newpop2 = self.mutate(newpop1, newpop2)
            
            #apply elitism and host tournaments
            if self.nelite > 0:
                newpop1, newpop2 = self.elitism(newpop1, newpop2, fitness)
                
            self.pop1, self.pop2 = newpop1, newpop2
            bestfit[i] = np.max(fitness)
                    
        print(np.concatenate((self.pop1, self.pop2), axis=1))
        final_fitness = self.fitfunc(self.pop1, self.pop2)
        print(final_fitness)
        best_index = np.argmax(final_fitness)
        
        self.best_index = best_index
        self.bestfit = bestfit
        self.best = np.concatenate((self.pop1[best_index], self.pop2[best_index]))
        
    def plot(self, plotval, title=None, xlabel=None, ylabel=None, popfunc=None, savefile=None):
        
        if plotval == 0:
            
            if title is None:
                title = "Maximum fitness vs. Generation count"
            if xlabel is None:
                xlabel = "Generation count"
            if ylabel is None:
                ylabel = "Max fitness in population"
            
            fig = plt.figure()
            plt.plot(np.arange(1,self.bestfit.shape[0] + 1), self.bestfit, '-rx', linewidth=4., markersize=15.)
            plt.xlabel(xlabel, fontsize=15.)
            plt.ylabel(ylabel, fontsize=15.)
            plt.suptitle(title, fontsize=18.)
            plt.show()
            
        elif plotval == 1:
            
            if popfunc is None:
                fittest = self.pop1[self.best_index]
            else:
                fittest = popfunc(self.pop1[self.best_index])
                
            ind = np.arange(fittest.shape[0])
            width = .35
            fig,ax = plt.subplots()
            ax.bar(ind + width, fittest, width, color='r')
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_title(title, fontsize=20)
            ax.set_xticks(ind + width*1.5)
            ax.set_yticks(np.arange(np.max(fittest)+2)+1)
            ax.set_xticklabels(np.arange(1,fittest.shape[0]+1))
            
        elif plotval == 2:
            
            if popfunc is None:
                fittest = self.pop2[self.best_index]
            else:
                fittest = popfunc(self.pop2[self.best_index])
                
            ind = np.arange(fittest.shape[0])
            width = .35
            fig,ax = plt.subplots()
            ax.bar(ind + width, fittest, width, color='r')
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_title(title, fontsize=20)
            ax.set_xticks(ind + width*1.5)
            ax.set_yticks(np.arange(np.max(fittest)+2)+1)
            ax.set_xticklabels(np.arange(1,fittest.shape[0]+1))
        
        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight')
    
        plt.show()
        
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
