#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:46:46 2022

@author: arbish
"""
import math
import numpy as np
from scipy import sparse

def create_mask(args):
        f = np.sqrt(args.f) 
        img_size = args.image_size
        d = img_size* img_size
        size_n = d+1
        brd = math.floor(f/2)
        masked = np.zeros((d,d))
        idx = 0
        inds1=[]
        for i in range(img_size):
            for j in range(img_size):
                inds2= []
                for m in range(i-brd,i+brd+1):
                    for n in range(j-brd,j+brd+1):
                        if m<0 or n<0:
                            continue
                        if m>=0 and m<img_size and n>=0 and n<img_size:
                             f = np.ravel_multi_index([m,n],(img_size,img_size),order='C')                                                               
                             inds1.append(f)
                    if len(inds1)!= 0:                        
                        inds2.append(inds1)
                    inds1=[]
                masked[idx,inds2]=1
                idx=idx+1  
          
        mask = np.zeros((d,d))
        mask[:,0:d] = masked
        M2 = sparse.csr_matrix(mask) 
        
        return M2
