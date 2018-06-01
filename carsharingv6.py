# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:01:59 2018

@author: Ibrahim
"""

import numpy as np
import itertools
import scipy.stats as stats
from scipy.stats import poisson, norm, randint
from scipy.integrate import quad
import pylab as pl
import random

# Number of stations
N=5
print("##########################################")
print("Total number of stations = " + str(N))
MAX_CARS=25
print("Max number of cars in the system = " + str(MAX_CARS))
print("##########################################")
# alphij : probability that a customer at station i goes to station j
#alph=np.random.rand(N,N)                                            #discontinued
#alph = np.apply_along_axis(lambda x: x / np.sum(x), 1, alph)        #discontinued
#print (alph)
###############################################################################
# configure expected demand functions for each station
###############################################################################
a=np.random.randint(30, 45, N)
a=a.astype(float)
#a=np.array([31., 38., 37., 31., 42.]) # fixed uncomment to generate a random vector
print("The expected demand model (deterministic)")
print("D(p)=a-b*p where a & b change randomly with")
print("every run - for fixed values set manually") 
print("a=" + str(a))
b=np.random.randint(-5, -1, N)
b=b.astype(float)
#b=np.array([-3., -4., -5., -5., -3.]) # fixed uncomment to generate a random vector
print("b="+ str(b))
pmin=np.empty(N); pmin.fill(3.0)
pmax=np.empty(N); pmax= a / (-1* b) -1 #set the price such that the expected demand cannot be negative
pmax=np.around(pmax, 1)
print("####################")
print("pmin=" + str(pmin))
print("pmax=" + str(pmax))
print("####################")
#expected demand function 
def D(p):
  for i in range(N):
   if p[i] >= pmax[i]:
      p[i]=pmax[i]
   elif p[i] <= pmin[i]:
      p[i]=pmin [i] 
  d=np.empty(N)
  d= a + b*p
  return d
#######################################
dmin=D(pmax)
dmax=D(pmin)
print("dmin=" + str(dmin))
print("dmax=" + str(dmax))
print("####################")
#######################################   
# inverse of expected demand function; returns the price for a given expected demand vector
def P(d):
  for i in range(N):
   if d[i] >= dmax[i]:
      d[i]=dmax[i]
   elif d[i] <= dmin[i]:
        d[i]=dmin[i] 
  #print (d)
  p=(d-a)/b
  p=np.around(p, 1)
 # print(p)
  return p
########################################
# Demand function
#low=-D(pmin) # the epsilon variable should be trucated at -D(p) to prevent negative demand
#low=low.astype(int)
#high=-low+1

def W(p):
 d = D(p)
 low=-D(p) # the epsilon variables should be trucated at -D(p) to prevent negative demand
 low=low.astype(int)
 high=-low+1
 #eps=np.random.randint(low[i],high[i],N)
 eps=[]
 for i in range(N):
     eps.append(np.random.randint(low[i],high[i]))
 #print(eps)
 #print(eps)
 #return np.maximum(d + eps,np.zeros(N))
 w=d + eps
 return w
#############################################
#testing 
#d=D(np.array([3,6,8,65,3], dtype=np.float))
#print("d="+ str(d))
#p=P(np.array([6, 10, 100, 20, 8], dtype=np.float))
#print("p=" + str(p))
#w=W(np.array([3,6,8,65,3], dtype=np.float))
#print("w="+str(w))
#############################################
#create all possible state vectors
print("####################")
print("Creating states vectors...")    
states= []
num = range(MAX_CARS + 1)
#for aCombination in itertools.product(num, repeat=N):
 #   states.append(aCombination)  
states = np.array([i for i in itertools.product(num, repeat=N) if sum(i)==MAX_CARS])   
print("States vectors created.")
print("Total no. of states = "+str(len(states)))
print("####################")
#############################################
#actions=[]
#for i in range(N):
#    actions.append(np.arange(dmin[i],dmax[i]+1))
##############################################
#policy = np.zeros(N)
policy=dmax
stateValue = np.zeros(len(states))
newStateValue = np.zeros(len(states))
improvePolicy = False
##############################################
low=-D(pmax)
low=low.astype(int)
high=-low+1
ranges=[]
for i in range(N):
    ranges.append(range(low[i],high[i]))
 
prob=1   
for i in range(N):
    prob *= 1./(high[i]-low[i]) 
 
lst = [list(x) for x in states]
disount_rate= 0.99 
##############################################################
# The below function is used instead of alphas as discussed
##############################################################
def randomize(n, num_terms):
    num_terms = num_terms  - 1
    a = np.random.randint(0, n, num_terms) 
    a=np.append(a,[0,n])
    a=np.sort(a)
    return [a[i+1] - a[i] for i in range(len(a) - 1)]
##############################################################
# This function returns the  value of being in a state 
# and taking a specific action     
##############################################################
def eval_policy(state, action):
    # initailize total return
    returns = 0.0
    discount_rate=0.99
    action=np.rint(action)
    action=action.astype(int)
    price=P(action)
    low=-D(price)
    low=low.astype(int)
    high=-low+1
    ranges=[]
    prob=1   
    for i in range(N):
        ranges.append(range(low[i],high[i]))
        prob *= 1./(high[i]-low[i]) 
    print("price="+str(price))
    n=0
    returns_prime=0
    flag=0
    while flag==0:
        epsVector=[]
        for i in range(N):
           epsVector.append(np.random.randint(low[i],high[i]))  
        newState=[]
        print("state="+str(state))
        print("action="+str(action))
        print("epsVector"+ str(epsVector))
        w=np.minimum(action + epsVector, state)
        temp=np.zeros(N)
        for j in range(N):
                w_j=w[j]
                w_j=w_j.astype(int)
                if w_j!=0:
                    temp +=randomize(w_j, N)
                else:
                    temp+=np.zeros(N)
        rewards =sum(w * price)
        print("reward="+str(rewards))
        newState=state+temp-w 
        #newState= newState.tolist()
        print("New state "+str(newState))
        state=newState
        #print(lst.index(newState))
#            returns +=prob * (rewards + disount_rate * stateValue[lst.index(newState)])
        returns_prime = returns
        returns += pow(discount_rate,n) * rewards #+ disount_rate * stateValue[lst.index(newState)])
        n += 1; print("n=" +str(n))
        if np.abs(returns - returns_prime) < 1e-5:
            flag=1
        print("returns=" +str(returns))
        print("###########################################")
    return returns
                            
rr= eval_policy(states[5], dmin) 



