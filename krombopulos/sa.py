
import numpy as np

def anneal(s0, neighbor, e, n, start_temp, 
           ts=20, p="boltzmann", schedule="linear"):
    """Return a low energy state through simulated annealing.
    
    Parameters:
    
        s0 (array-like):
            The initial state of the annealing algorithm
        neighbor (func):
            Function that returns a 'neighboring' state from input state
        e (func):
            Energy function of an input state.
        n (int):
            Number of iterations in the annealing algorithm.  Lower numbers
              correspond to higher annealing rates.
        start_temp (float):
            The starting temperature in the annealing schedule.
            
    Keywords:
    
        ts (int):
            Swing value.  The annealing halts if the solution hasn't changed
              in 'ts' iterations.  Defaults to 20.
        p ({'boltzmann'}):
            Acceptance probability function.  Defaults to 'boltzmann'.
        schedule ({'linear', 'exponential'}):
            Annealing schedule.  Defaults to linear."""
            
    if p.lower() == "boltzmann":
        p = prob_boltzmann
    else:
        raise ValueError("Invalid acceptance probability function")
        
    if schedule.lower() == "linear":
        schedule = sch_linear(start_temp)
    elif schedule.lower() == "exponential":
        schedule = sch_exponential(start_temp)
    else:
        raise ValueError("Invalid annealing schedule")
        
    e0 = e(s0)
    sbest = s0
    ebest = e0
    sList, eList = [s0], [e0]
    
    stopcount = 0
    
    for i in range(n-1):
        
        temp = schedule(i/n)
        s1 = neighbor(np.copy(s0))
        e1 = e(s1)
        
        if e1 < ebest:
            sbest = np.copy(s1)
            ebest = e1
            
        ediff = e1 - e0
        if np.abs(ediff) < 1e-6:
            stopcount += 1
            
            if stopcount >= ts:
                print("Stopping condition met")
                break
        else:
            stopcount = 0
        
        if p(e0,e1,temp) + np.random.rand() >= 1.:
            e0 = e1
            s0 = s1
            
        sList.append(s0)
        eList.append(e0)
    
    s1 = neighbor(np.copy(s0))
    e1 = e(s1)
    
    if e1 < ebest:
        sbest = s1
        ebest = e1
    
    if e1 < e0:
        s0 = s1
        e0 = e1
        
    sList.append(s0)
    eList.append(e0)
        
    print('best e: {}'.format(ebest))
    print('best s: {}'.format(sbest))
    print('final e: {}'.format(e0))
    print('final s: {}'.format(s0))

    return s0, np.array(eList), np.array(sList), sbest, ebest
            
def prob_boltzmann(e0, e1, temp):
    return np.exp((e0-e1)/temp)
    
def sch_linear(start_temp):
    
    def schedule(ratio):
        return start_temp*(1. - ratio)
        
    return schedule
    
def sch_exponential(start_temp):
    
    alpha = 0.99
    rate = 1. - alpha    
    
    def schedule(ratio):
        return start_temp*np.power(rate, ratio)
        
    return schedule