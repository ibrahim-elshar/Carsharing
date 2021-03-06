# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:01:59 2018

@author: Ibrahim
"""

import numpy as np
import itertools
import timeit

# Number of stations
N=2
print("##########################################")
print("Total number of stations = " + str(N))
MAX_CARS=30
print("Max number of cars in the system = " + str(MAX_CARS))
print("##########################################")
###############################################################################
# configure expected demand functions for each station
###############################################################################
a=np.random.randint(30, 45, N).astype(float)
#a=a.astype(float)
#a=np.array([31., 38., 37., 31., 42.]) # fixed uncomment to generate a random vector
print("The expected demand model (deterministic)")
print("D(p)=a+b*p where a & b change randomly with")
print("every run - for fixed values set manually") 
print("a=" + str(a))
b=np.random.randint(-5, -1, N).astype(float)
#b=b.astype(float)
#b=np.array([-3., -4., -5., -5., -3.]) # fixed uncomment to generate a random vector
print("b="+ str(b))
pmin=np.empty(N); pmin.fill(3.0)
pmax=np.empty(N); pmax= a / (-1* b) -0.1 # -1 set the price such that the expected demand cannot be negative
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
  ######
  d=np.rint(d).astype(int)
  ######
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
vard="y"###################################################
if vard=="y":
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
          
############################################
actions=[]
#for i in range(N):
#    actions.append(np.arange(dmin[i],dmax[i]+1))
ranges=[]
prob=1   
for i in range(N):
    ranges.append(range(dmin[i],dmax[i]))
actions=list(itertools.product(*ranges))
##############################################################
# The below function is used instead of alphas as discussed
##############################################################
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
##############################################################
# This function returns the  value of being in a state 
# and taking a specific action     
##############################################################
##############################################################
# This function returns the  value of being in a state 
# and taking a specific action     
##############################################################
T=np.random.randint(1,4, size=(N, N))
#T=np.ones((N, N)).astype(int)
kmax=np.max(T)
Nik= []
for i in range(N):
    Nik.append([np.where(T[:,i] == k) for k in range(1, kmax+1)])
##############################################################
def eval_policy(state, s, price, t ):
    # initailize total return
    returns = 0.0
    discount_rate=0.99
#    price=P(action)
#    demand=D(price)
#    demand=np.rint(demand)
#    demand=demand.astype(int)
    demand=np.rint(D(price)).astype(int)
    low=-D(price)
    low=low.astype(int)
    high=-low+1
    n=0
    returns_prime=0
    flag=0
    while flag==0:
        epsVector=[]
        for i in range(N):
           epsVector.append(np.random.randint(low[i],high[i])) 
        newState=[]
        print("state="+str(state))
        print("s="+str(s))
        print("price="+str(price))
        print("demand="+str(demand))
        print("epsVector"+ str(epsVector))
        w=np.minimum(demand + epsVector, state)
        print("w="+str(w))
        wij=np.zeros((N,N))
        for j in range(N):
#                print("w[j]="+str(w[j]))
                if w[j]!=0:
                    wij[j,:]=randomize(w[j], N) #wij=[w_j1 w_j2 w_j3 w_j4 ... w_ji... w_jN]
#                    print("wij["+str(j)+",:]="+str(wij[j,:]))
                else:
                    wij[j,:]=np.zeros(N)
#                    print("wij["+str(j)+",:]="+str(wij[j,:]))
        print("wij="+str(wij))
#        print("T="+str(T))
        Twij=np.multiply(T, wij)    
#        print("Twij="+str(Twij))
#        print(np.sum(Twij, axis=1))
#        print("price="+str(price))
        rewards =sum(np.sum(Twij, axis=1) * price)
        print("reward="+str(rewards))
        temp=np.zeros(N)
        for i in range(N):
            temp[i]=np.sum([wij[j,i] for j in Nik[i][0]])
        newState=state+s[:,0] +temp -w 
        print("New state "+str(newState))
        newS=np.zeros((N,kmax))
        if kmax>1:
            for i in range(N):
                for k in range(kmax):
                    if k==0:
                        newS[i,k]= s[i,k+1] 
                    elif k==(kmax-1):
                        newS[i,k]=np.sum([wij[j,i] for j in Nik[i][k]])
                    else:   
                        newS[i,k]=np.sum([wij[j,i] for j in Nik[i][k]]) + s[i,k+1]
        print("newS="+str(newS))
        state=newState
        s=newS
#        returns_prime = returns
        returns += pow(discount_rate,n) * (rewards + 0.00001) #+ reward is perturbed to avoid stopping at the case where the reward is zero
        print("t=" +str(n)) 
        n += 1
        if n==t-1:#np.abs(returns - returns_prime) < 1e-10:
            flag=1
        print("returns=" +str(returns))
        print("###########################################")
    return returns


start_time = timeit.default_timer()
rrr= eval_policy(randomize(MAX_CARS,N),np.zeros((N,kmax)), pmin, 12) 
elapsed = timeit.default_timer() - start_time                           
print("Time="+str(elapsed))


