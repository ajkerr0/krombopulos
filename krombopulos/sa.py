"""


@author: Alex Kerr
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.close("all")

#define system
SIZE = 20
SPINCHOICE = [-1,1]
JCHOICE = [-1,1]
JS = np.random.choice(JCHOICE, size=SIZE-1)
#JS = np.array([-1,1,1,1,1,-1,-1,1,-1])

def energy(state):
    
    e = 0.
    for index,j in zip(np.arange(SIZE-1),JS):
        e += j*(state[index]*state[index+1])   
    return e
    
def neighbor(state):
    """Return a neighbor state of the input state."""
    
    index = np.random.randint(0, high=SIZE)
#    print(index)
    newstate = np.copy(state)
    newstate[index] *= -1
    return newstate
    
def temperature(ratio):
    """Return temperature as a function of fractional progress"""
#    print(1-ratio)
    return 5.*(1. - ratio)
    
def p(e0,e1,T):
    return np.exp((e0-e1)/T)
    
def sa(n=250):
    
    s0 = np.random.choice(SPINCHOICE, size=SIZE)
#    print('first state')
#    print(s0)
    e0 = energy(s0)
    sbest = s0
    ebest = e0
#    print(e0)
    eList = [e0]
    sList = [s0]
    
    for i in range(n):
        T = temperature(i/n)
#        print(T)
        s1 = neighbor(s0)
        e1 = energy(s1)
        
        if e1 < ebest:
            ebest = e1
            sbest = s1
        
        if p(e0,e1,T) + np.random.rand() >= 1.:
#            print('change')
            e0 = e1
            s0 = s1
            
        eList.append(e0)
        sList.append(s0)
            
    s1 = neighbor(s0)
    e1 = energy(s1)
    
    if e1 < ebest:
        ebest = e1
        sbest = s1
            
    if e1 < e0:
        s0 = s1
        
    print('best: {}'.format(ebest))
    print(sbest)
    
    return s0,eList,np.array(sList)
    
#analytical solution
sol1 = [-1]
val = -1

for j in JS:
    sol1.append(j*val)

sol1 = np.array(sol1)
sol2 = -sol1
sol = np.array((sol1,sol2))

TIME=20
xspace=5
yspace=5
xlabels = ['']*SIZE
ylabels = ['']*TIME

for count in range(len(xlabels)):
    if count%xspace == 0:
        xlabels[count] = count
xlabels.append(SIZE)
        
for count in range(len(ylabels)):
    if count%yspace == 0:
        ylabels[count] = count
ylabels.append(TIME)
ylabels.reverse()
    
print(JS)    
for i in range(1):
    n = 250
    s,eList,sList = sa(n)
    fig = plt.figure()
    
    #energy plot
    plt.plot(np.arange(len(eList)),eList, '-kx')
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.suptitle("Energy vs. Iteration", fontsize=20)
    plt.show()
    
    #state plot
    
    fig2 = plt.figure()
    gs = gridspec.GridSpec(4,1, height_ratios=[1, TIME//2, TIME//2, TIME//2])
    
    a1 = plt.subplot(gs[0])
    plt.axis([0,SIZE,0,2])
    plt.xticks(np.arange(0,SIZE+1), ['']*SIZE)
    plt.yticks([0,1], ['',''])
    plt.tick_params(axis='both', top='off', which="both", bottom="off", right="off", left="off")
    plt.grid(True, color='gray', linestyle='-', linewidth=1)
    plt.imshow(sol,cmap="Greys", interpolation='nearest', extent=[0,SIZE,0,2])
    
    trange = (0,20)
    ylabels = ['']*TIME            
    for count in range(len(ylabels)):
        if count%yspace == 0:
            ylabels[count] = count + trange[0]
    ylabels.append(TIME + trange[0])
    ylabels.reverse()
    a2 = plt.subplot(gs[1])
    plt.imshow(sList[trange[0]:trange[1]], cmap="Greys", interpolation='nearest', extent=[0,SIZE,0,TIME])
    plt.axis([0,SIZE,trange[0],trange[1]])
    plt.xticks(np.arange(0,SIZE+1), ['']*SIZE)
    plt.yticks(np.arange(trange[0],trange[1]+1), ylabels[:TIME])
    plt.tick_params(axis='both', top='off', which="both", bottom="off", right="off", left="off")
    plt.grid(True, color='gray', linestyle='-', linewidth=1)
    plt.tight_layout()
    
    trange = (100,120)
    ylabels = ['']*TIME            
    for count in range(len(ylabels)):
        if count%yspace == 0:
            ylabels[count] = count + trange[0]
    ylabels.append(TIME + trange[0])
    ylabels.reverse()
    a3 = plt.subplot(gs[2])
    plt.imshow(sList[trange[0]:trange[1]], cmap="Greys", interpolation='nearest', extent=[0,SIZE,0,TIME])
    plt.axis([0,SIZE,trange[0],trange[1]])
    plt.xticks(np.arange(0,SIZE+1), ['']*SIZE)
    plt.yticks(np.arange(trange[0],trange[1]+1), ylabels[:TIME])
    plt.tick_params(axis='both', top='off', which="both", bottom="off", right="off", left="off")
    plt.grid(True, color='gray', linestyle='-', linewidth=1)
    plt.tight_layout()
    
    trange = (230,250)
    ylabels = ['']*TIME            
    for count in range(len(ylabels)):
        if count%yspace == 0:
            ylabels[count] = count + trange[0]
    ylabels.append(TIME + trange[0])
    ylabels.reverse()
    a4 = plt.subplot(gs[3])
    plt.imshow(sList[trange[0]:trange[1]], cmap="Greys", interpolation='nearest', extent=[0,SIZE,0,TIME])
    plt.axis([0,SIZE,trange[0],trange[1]])
    plt.xticks(np.arange(0,SIZE+1), xlabels)
    plt.yticks(np.arange(trange[0],trange[1]+1), ylabels[TIME:])
    plt.tick_params(axis='both', top='off', which="both", bottom="off", right="off", left="off")
    plt.grid(True, color='gray', linestyle='-', linewidth=1)
    plt.tight_layout()
