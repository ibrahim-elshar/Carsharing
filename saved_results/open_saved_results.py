# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:12:50 2018

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
MAX_CARS_def=10
num_stages_def = 6
pmin_def=np.array([1., 1.])
pmax_def=np.array([10000., 10000.])
#a_def=np.random.randint(30, 45, N_def).astype(float)
#b_def=np.random.randint(-5, -1, N_def).astype(float)
a_def=np.array([20., 20.])
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
    
    
    def val1(self, action, state, stateValue, t): # a car rented from a station will go to the other station w.p 1
        assert self.contains(action)
#        print("action="+str(action))
        price=self.PA(action)
        returns = 0.0
        low=-np.array([action, self.dB]).astype(int)
#        print("price="+str(price))
        high=-low+1
#        print("low="+str(low))
#        print("high="+str(high))
        ranges=[]
        prob=1   
        for i in range(self.N):
            ranges.append(range(low[i],high[i]))
            prob *= 1./(high[i]-low[i]) 
#        print("ranges="+str(ranges))
#        print("prob="+str(prob))
        v=list(itertools.product(*ranges))
        bi_state=np.array([state, self.MAX_CARS-state])
        bi_price=np.array([price, self.pB])
        bi_action=np.array([action, self.dB])
#        print("bi-price="+str(bi_price))
#        print("bi_state="+str(bi_state))
#        print("bi_action="+str(bi_action))
#        print(v)
        for i in range(len(v)):
                epsVector=v[i]
                w=np.minimum( bi_action + epsVector, bi_state); #print("w="+str(w))
                reward =sum(w * bi_price)
#                print("reward="+str(reward))
#                newState = (state +temp[0] - w[0]).astype(int)
                newState = state + w[1] -w[0]
#                print("newState="+str(newState))
                returns +=prob * (reward + self.discount_rate * stateValue[t, newState])
#                print("returns="+str(returns))
#                print("#################################")
        return returns

    def val2(self, action, state, stateValue, t): # a car rented from a station can go back to the same station at the end o period
        assert self.contains(action)
#        print("action="+str(action))
        price=self.PA(action)
        returns = 0.0
        low=-np.array([action, self.dB]).astype(int)
#        print("price="+str(price))
        high=-low+1
#        print("low="+str(low))
#        print("high="+str(high))
        ranges=[]
        prob=1   
        for i in range(self.N):
            ranges.append(range(low[i],high[i]))
            prob *= 1./(high[i]-low[i]) 
#        print("ranges="+str(ranges))
#        print("prob="+str(prob))
        v=list(itertools.product(*ranges))
        bi_state=np.array([state, self.MAX_CARS-state])
        bi_price=np.array([price, self.pB])
        bi_action=np.array([action, self.dB])
#        print("bi-price="+str(bi_price))
#        print("bi_state="+str(bi_state))
#        print("bi_action="+str(bi_action))
#        print(v)
        for i in range(len(v)):
                epsVector=v[i]
#                print("epsVector="+str(epsVector))
                w=np.minimum( bi_action + epsVector, bi_state); #print("w="+str(w)) 
                w1=np.array([i for i in itertools.product(range(w[0]+1), repeat=2) if sum(i)==w[0]]) 
                w2=np.array([i for i in itertools.product(range(w[1]+1), repeat=2) if sum(i)==w[1]])
                probw=1./(len(w1) *len(w2))
                for i1 in range(len(w1)):
                    for i2 in range(len(w2)):
                        temp=w1[i1]+w2[i2]                         
        #                print("temp="+str(temp))
                        reward =sum(w * bi_price)
#                        print("reward="+str(reward))
                        newState = (state +temp[0] - w[0]).astype(int)
#                        print("newState="+str(newState))
                        returns +=prob * probw *(reward + self.discount_rate * stateValue[t, newState])
#                        print("returns="+str(returns))
        #                print("#################################")
        return returns
    def val3(self, action, state, stateValue, t): # a car rented from a station can go back to the same station at the end o period
        assert self.contains(action)
#        print("action="+str(action))
        price=self.PA(action)
        returns = 0.0
        low=-np.array([action, self.dB]).astype(int)
#        print("price="+str(price))
        high=-low+1
#        print("low="+str(low))
#        print("high="+str(high))
        ranges=[]
        prob=1   
        for i in range(self.N):
            ranges.append(range(low[i],high[i]))
            prob *= 1./(high[i]-low[i]) 
#        print("ranges="+str(ranges))
#        print("prob="+str(prob))
        v=list(itertools.product(*ranges))
        bi_state=np.array([state, self.MAX_CARS-state])
        bi_price=np.array([price, self.pB])
        bi_action=np.array([action, self.dB])
#        print("bi-price="+str(bi_price))
#        print("bi_state="+str(bi_state))
#        print("bi_action="+str(bi_action))
#        print(v)
        for i in range(len(v)):
                epsVector=v[i]
#                print("epsVector="+str(epsVector))
                w=np.minimum( bi_action + epsVector, bi_state); #print("w="+str(w)) 
                w1=np.array([i for i in itertools.product(range(w[0]+1), repeat=2) if sum(i)==w[0]]) 
                w2=np.array([i for i in itertools.product(range(w[1]+1), repeat=2) if sum(i)==w[1]])
                bn1 = ss.binom(w[0], self.prob_ij[0,0])
                bn2 = ss.binom(w[1], self.prob_ij[1,1])
                for i1 in range(len(w1)):
                    for i2 in range(len(w2)):
                        temp=w1[i1]+w2[i2]  
                        probw= bn1.pmf(w1[i1,0]) * bn2.pmf(w2[i2,1])
        #                print("temp="+str(temp))
                        reward =sum(w * bi_price)
#                        print("reward="+str(reward))
                        newState = (state +temp[0] - w[0]).astype(int)
#                        print("newState="+str(newState))
                        returns +=prob * probw *(reward + self.discount_rate * stateValue[t, newState])
#                        print("returns="+str(returns))
        #                print("#################################")
        return returns
    
    
    def solve(self):

        for t in range(self.num_stages - 2, -1, -1):
        #    print("t="+str(t))
            for state in self.states:
        #        print("state="+str(state))
                value_action = {}
                for action in self.actions: #allowable_actions[t, n]:
        #            print("action="+str(action))
                    value_action[action] = self.val3(action, state,value, t+1)
        #            print("value_action="+str(value_action[self.alst.index(action.tolist())]))
                value[t, state] = max(value_action.values())
        #        print("v("+str(state)+")="+str(value[t, self.slst.index(state_index)]))
                policy[t, state] =  list (
                x for x in self.actions
                #if isclose(value_action[x], value[t,state])
                if np.sum(np.abs(value_action[x] - value[t,state])) < 1e-5
                )
        #        print("Policy="+str( policy[t, self.slst.index(state_index)]))
        return value, policy
    

def printValue(num_stages, x, data, labels):
    global figureIndex
    figureIndex += 1  
    for t in range(num_stages-1):
        fig, ax = plt.subplots()
#        plt.figure(figureIndex)  
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
#        plt.figure(figureIndex)  
        ax.scatter(x, [data[t, i][0] for i in range(len(x))])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(labels[2]+ str(t))
        for i in range(len(x)):
                    ax.annotate((x[i] , data[t, i][0]), (x[i] , data[t, i][0])) 
        plt.show()
        figureIndex += 1            
    
class StageStateData(dict):
    def __init__(self, number_of_stages, states):
        self.number_of_stages = number_of_stages
        self.states = states

    def __repr__(self):
        entries = []
        for (t, n), v in sorted(self.items()):
            entries.append(" (stage: {0}, state: {1}): {2}".format(t, self.states[n], v))
        string = '{' + '\n'.join(entries).strip() + '}'
        return string   

# Getting back the objects:
#cwd = os.getcwd()
#ofilename=(cwd+"\\saved_results\\2Stations_10MC_6T_10.0-20.0a_-5.0-5.0 b_0.1-0.95P")  
ofilename=("2Stations_10MC_6T_20.0-20.0a_-5.0-5.0 b_0.1-0.95P")  
with open(str(ofilename)+".pkl", 'rb') as f:  # Python 3: open(..., 'rb')
    env, value, policy, elapsed_time = pickle.load(f)
#    env, value, policy = pickle.load(f)


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
#print(value)
#print(policy)      

#plt.scatter(env.states,[policy[0, i][0] for i in range(len(env.states))])
#plt.show()
figureIndex = 0
printValue(2, env.states,value, ["# of cars in first location","Expected returns","Expected returns in stage "]) #env.num_stages
printPolicy(2, env.states,policy, ["# of cars in first location","action"," Demand policy of stage "]) #env.num_stages                                             
plt.show()
dd=np.zeros(len(env.states))
for i in env.states:
    dd[i]=(policy[0,i][0])
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

