# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:05:55 2018

@author: Ibrahim
"""

import numpy as np
import itertools
import timeit
import matplotlib.pyplot as plt
import scipy.stats as ss
import os
import pickle
#%matplotlib auto
N_def=2
MAX_CARS_def=20
num_stages_def = 12
pmin_def=np.array([1., 1.])
pmax_def=np.array([10000., 10000.])
#a_def=np.random.randint(30, 45, N_def).astype(float)
#b_def=np.random.randint(-5, -1, N_def).astype(float)
a_def=np.array([25., 25.])
b_def=np.array([-5., -5.])
discount_rate_def=0.99
prob_ij_def=np.array([[0.1, 0.9],[0.05, 0.95]])  #probability of a customer going from station i to j
def randomize(n, num_terms):
    if n == 1:
        a=np.zeros(num_terms).astype(int)
        i=np.random.randint(0, num_terms)
        a[i]=1
        return [a[x] for x in range(num_terms)]
    else:
        num_terms = num_terms  - 1
        a = np.random.randint(0, n, num_terms) 
        a=np.append(a,[0,n])
        a=np.sort(a)
        return [a[i+1] - a[i] for i in range(len(a) - 1)]

def rang(x):
   ends = np.cumsum(x)
#   print("ends="+str(ends))
   ranges = np.arange(ends[-1])
#   print("ranges="+str(ranges))
#   print("ends-x"+str(ends-x))
#   print("np.repeat(ends-x, x)="+str(np.repeat(ends-x, x)))
   rangess = ranges - np.repeat(ends-x, x)
#   print("rangess="+str(rangess))
   ww=np.repeat(x-1, x)
#   print(ww)
   return rangess, ww           


class CarsharingEnv():
    
    def __init__(self, num_stations=N_def, num_cars=MAX_CARS_def, min_price=pmin_def, max_price=pmax_def, a=a_def, b=b_def, discount_rate=discount_rate_def, num_stages=num_stages_def,prob_ij=prob_ij_def):
        self.N = num_stations  # Number of stations
        self.MAX_CARS = num_cars # Max number of cars in the system 
        self.set_pmin= min_price
        self.a=a
        self.b=b
        self.pmin=min_price
        self.pmax=np.minimum(np.around(self.a / (-1* self.b) -0.1, 1), max_price)  # -0.1 set the price such that the expected demand cannot be negative and minimum between the max_set_price and the one computed  
        self.dmin=self.D(self.pmax)
        self.dmax=self.D(self.pmin)
        self.dAmin=self.dmin[0]
        self.dAmax=self.dmax[0]
        self.pB=(self.pmin[1]+self.pmax[1])*0.5
        self.dB=self.DB(self.pB)     
        self.actions=np.array(range(self.dAmin,self.dAmax+1))
        self.states =np.array(range(self.MAX_CARS+1))
        self.num_stages=num_stages_def
        self.discount_rate=discount_rate
        self.prob_ij=prob_ij
        
    #expected demand function 
    def D(self, p):
      d= np.rint(self.a + self.b*p).astype(int)
      return d
  
    def DA(self, p): 
      d= np.rint(self.a[0] + self.b[0]*p).astype(int)
      return d
  
    def DB(self, p): 
      d= np.rint(self.a[1] + self.b[1]*p).astype(int)
      return d
  
    # inverse of expected demand function; returns the price for a given expected demand vector
    def P(self, d):
      p=np.around((d-self.a)/self.b, 1)
      return p
  
    def PA(self, d): 
      p=np.around((d-self.a[0])/self.b[0], 1)
      return p  
  
    def PB(self, d): 
      p=np.around((d-self.a[1])/self.b[1], 1)
      return p  
  
    def contains(self, x):
        return (x >= self.dAmin).all() and (x <= self.dAmax).all()
    
    
    def Exp_Val(self, action, state, stateValue, t): # a car rented from a station can go back to the same station at the end o period
        assert self.contains(action)
        price=self.PA(action)
        returns = 0.0
        low=-np.array([action, self.dB]).astype(int)
        high=-low+1
        ranges=[]
        probEps=1   
        for i in range(self.N):
            ranges.append(range(low[i],high[i]))
            probEps *= 1./(high[i]-low[i]) 
        v1=np.array(list(ranges[0]))
        v2=np.array(list(ranges[1]))
        wA=np.minimum( action + v1, state)
        wB=np.minimum( self.dB + v2, self.MAX_CARS-state)
        w1,wArp=rang(wA+1)
        w2,wBrp=rang(wB+1)
        temp=w1[:, None] + w2
        temp = temp.ravel()
        b1 = ss.binom.pmf(w1, wArp, self.prob_ij[0,0])
        b2 = ss.binom.pmf(w2, wBrp,  self.prob_ij[1,0])
        prob_array = b1[:, None] * b2 *probEps
        prob = prob_array.ravel()
        wArpp=np.repeat(wArp, len(w2))
        newState =temp + state -wArpp
        reward= (wArp*price)[:, None] + (wBrp*env.pB) 
        reward=reward.ravel()
        ar=stateValue[t,:]
        vl=ar[newState]
#        vl=stateValue[t,newState]
        returns = sum(prob*list(reward + self.discount_rate * vl))
        return returns    
    
    def solve(self):       
        for t in range(self.num_stages - 2, -1, -1):
            for state in self.states:
                value_action = {}
                for action in self.actions: 
                    if (state>0) and (action < min(policy[t, state-1])): 
                        value_action[action]=0
                        continue
                    value_action[action] = self.Exp_Val(action, state,value, t+1)
                value[t, state] = max(value_action.values())
                policy[t, state] =  list (
                x for x in self.actions
#                if isclose(value_action[x], value[t,state])
                if np.sum(np.abs(value_action[x] - value[t,state])) < 1e-5
                )
        return value, policy
    

def printValue(num_stages, x, data, labels):
    global figureIndex
    figureIndex += 1  
    for t in range(num_stages-1):
        fig, ax = plt.subplots()
        ax.scatter(x, [data[t, i] for i in range(len(x))])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(labels[2]+ str(t))
        plt.show()
        figureIndex += 1
def printPolicy(num_stages, x, data, labels):
    global figureIndex
    figureIndex += 1  
    for t in range(num_stages-1):
        fig, ax = plt.subplots()
        ax.scatter(x, [min(data[t, i]) for i in range(len(x))])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(labels[2]+ str(t))
        for i in range(len(x)):
            ax.annotate((x[i] , min(data[t, i])), (x[i] , min(data[t, i]))) 
        plt.show()
        figureIndex += 1            
    
class StageStateData(dict):
    def __init__(self, number_of_stages, states):
        self.number_of_stages = number_of_stages
        self.states = set(states)

    def __setitem__(self, key, value):
        try:
            t, n = key
        except ValueError:
            if len(key) != 2:
                raise ValueError('Incorrect number of indices')
            else:
                raise

        if ((n not in self.states) or
                (t not in range(self.number_of_stages))):
            raise KeyError('Invalid stage or state')
        super().__setitem__(key, value)

    def __getitem__(self, key):
        try:
            t, n = key
        except ValueError:
            if len(key) != 2:
                raise ValueError('Incorrect number of indices')
            else:
                raise
        if isinstance(n, slice):  
             return np.array([self[ii] for ii in zip(itertools.repeat(t),range(env.MAX_CARS+1))])
        elif isinstance(n, np.ndarray):
            return np.array([self[ii] for ii in zip(itertools.repeat(t),n)])
        else:
            return super().__getitem__(key)

    def __repr__(self):
        entries = []
        for (t, n), v in sorted(self.items()):
            entries.append(" (stage: {0}, state: {1}): {2}".format(t, n, v))
        string = '{' + '\n'.join(entries).strip() + '}'
        return string


             
env=CarsharingEnv()
print("##########################################")
print("Total number of stations = " + str(env.N))
print("Max number of cars in the system = " + str(env.MAX_CARS))
print('A customer stays at station A w.p {0}'.format(env.prob_ij[0,0]))
print('A customer stays at station B w.p {0}'.format(env.prob_ij[1,1]))
print("Number of stages="+str(env.num_stages))
print("Discount rate="+str(env.discount_rate))
print("##########################################")
print("The expected demand model (deterministic)")
print("D(p)=a+b*p where a & b change randomly with")
print("every run - for fixed values set manually") 
print("a=" + str(env.a))
print("b="+ str(env.b))
print("####################")
print("pmin=" + str(env.pmin))
print("pmax=" + str(env.pmax))
print("dmin=" + str(env.dmin))
print("dmax=" + str(env.dmax))
print("pB=" + str(env.pB))
print("dB=" + str(env.dB))
print("####################")
value = StageStateData(env.num_stages, env.states)
policy = StageStateData(env.num_stages, env.states)
for i in  range(len(env.states)):
    value[env.num_stages-1, i]=0
#value = np.zeros((env.num_stages, env.MAX_CARS+1))
#policy = np.zeros((env.num_stages, env.MAX_CARS+1))
start_time = timeit.default_timer()
value, policy=env.solve()
elapsed_time = timeit.default_timer() - start_time
print("Time="+str(elapsed_time))
#print(value)
#print(policy)      
figureIndex = 0
printValue(2, env.states,value, ["# of cars in first location","Expected returns","Expected returns in stage "]) #env.num_stages
printPolicy(2, env.states,policy, ["# of cars in first location","action"," Demand policy of stage "]) #env.num_stages                                             
plt.show()
dd=np.zeros(len(env.states))
for i in env.states:
    dd[i]=(min(policy[0,i]))
pr=np.zeros(len(env.states))
for i in env.states:
    pr[i]=env.PA(dd[i])
#plt.scatter(env.states,pr) 
fig, ax = plt.subplots()
#plt.figure(figureIndex)  
ax.scatter(env.states,pr) 
ax.set_xlabel("# of cars in first location")
ax.set_ylabel("Price of renting a car in 1st location")
ax.set_title("Price policy of stage 0")
for i in range(len(env.states)):
    ax.annotate((env.states[i] , pr[i]), (env.states[i] , pr[i])) 

## First check solve() for which val is set
#%matplotlib auto # copy %matplotlib and paste in the python console so the plots will be printed in a separate window.
#%matplotlib inline
############################ val1
############################ val2


# Saving the objects:
cwd = os.getcwd() # gets current working directory from os
filename=(cwd+"\\saved_results\\"+str(env.N) + "Stations_"+str(env.MAX_CARS)+"MC_" +str(env.num_stages)+"T_"+str(env.a[0])+"-" +str(env.a[1])+"a_"+str(env.b[0])+str(env.b[1])+" b_"+str(env.prob_ij[0,0])+"-" +str(env.prob_ij[1,1])+"P") 
with open(str(filename)+".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([env, value, policy, elapsed_time], f)

## Getting back the objects:
#cwd = os.getcwd()
#ofilename=(cwd+"\\saved_results\\2Stations_10MC_6T_10.0-20.0a_-5.0-5.0 b_0.1-0.95P")  
#with open(str(ofilename)+".pkl", 'rb') as f:  # Python 3: open(..., 'rb')
#    env, value, policy = pickle.load(f)