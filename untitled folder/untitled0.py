# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 06:52:41 2018

@author: Ibrahim
"""
import numpy as np

def st(B):
     # Sort B based on first row
    sB = B[:,np.argsort(B[0,:])]

    # col mask of where each group ends
    col_mask = np.append(np.diff(sB[0,:],axis=0)!=0,[True])

    # Get cummulative summations and then DIFF to get summations for each group
    cumsum_grps = sB.cumsum(1)[1:,col_mask]
    sum_grps = np.diff(cumsum_grps,axis=1)

    # Concatenate the first unique col with its counts
    counts = np.concatenate((cumsum_grps[:,0][None].T,sum_grps),axis=1)

    # Concatenate the first row of the input array for final output
    out = np.concatenate((sB[0,col_mask][None,:],counts),axis=0)
    return out