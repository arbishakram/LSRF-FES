#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:46:46 2022

@author: arbish
"""

import numpy as np
from scipy import sparse
import random
from sklearn.preprocessing.data import normalize as f_normalize
from joblib import Parallel, delayed

# Normalize each feature in the input matrix.
def normalize_X(self, M1, M2): 
     Xtt = M1[:, :self.image_size, :]     
     le = Xtt.shape[0]
     norm_Xtt = np.zeros((le, self.image_size, 3))    
     X_offset = np.average(Xtt, axis=0)     
     Xtt = Xtt - X_offset    
     for ch in range(self.ch):
         tXtt, X_scale = f_normalize(Xtt[:,:,ch], axis=0, copy=False, return_norm=True)   
         norm_Xtt[:,:,ch] = tXtt        
     return norm_Xtt
 
# Initialize the locality constraint with 9-neighbors.
def initialize_local9(self, lamd_t, Mk):
     indices = []
     length = []
     for m in range(self.image_size):   
             ind = Mk[m].indices
             indices.append(ind)             
     for k in range(self.image_size):              
          ind = indices[k]
          l = len(ind)
          lamd_t[k, 0:l, 0] = ind    
          length.append(len(ind))
     return lamd_t, length
          
          
# Initialize the locality constraint with 5-neighbors.       
def initialize_local5(self, lamd_t, Mk):
     indices = []
     length = []
     for m in range(self.image_size):   
             ind = Mk[m].indices
             indices.append(ind)             
     for k in range(self.image_size):
         ind = indices[k]   
         l = len(ind)
         if l == 4:             
             lamd_t[k, 0:l, 0] = ind
             length.append(len(ind))
         elif l == 6:
             li = len(ind[1::2])
             lamd_t[k, 0:li, 0] = ind[1::2]  
             length.append(len(ind[1::2]))
         else:
             lamd_t[k, :, 0] = ind[0::2]
             length.append(len(ind[0::2]))
     return lamd_t, length
    
    
# Initialize the locality constraint randomly.   
def initialize_random(self, lamd_t):
    for k in range(self.f):
          lamd_t[:,k,0] = random.sample(range(0, self.image_size), self.image_size)
    return lamd_t
 

def omp_with_locality_constraint(self, local_neighbors, Mk, M1, M2):         
     
     norm_Xtt = normalize_X(self, M1, M2)
     local_neighbors_list = []
     for m in range(self.image_size):   
             ind = local_neighbors[m].indices
             local_neighbors_list.append(ind)   
             
               
     lamd_t = np.zeros((self.image_size, self.f, 1))      # maximum 2
#     lamd_t = initialize_random(self, lamd_t)
     lamd_t, length  = initialize_local9(self,lamd_t, Mk)

     for tau in range(1):
        W = np.zeros((self.image_size, self.size_n))                 
        def compute_wi(k):
#        for k in range(self.image_size): 
                  print(k)
                  tk = M2[:, k, :]
                  r =  tk                   
                  ind = set()
                  for f in range(self.f): 
                      t_ind = 0
                      ### compute feature-space similarity for all pixels
                      for ch in range(3):
                         t_ind +=  np.dot(norm_Xtt[:, :, ch].T, r[:, ch])   
                      d = np.zeros(self.image_size)                                       
                      for j in range(self.image_size):  
                          if j in ind:
                              continue                                            
                          jloc = np.array(np.unravel_index(j, (self.size, self.size)))                                              
                          dist = 0
                          
                          ### compute spatial similarity 
                          for idx, m in enumerate(local_neighbors_list[k]):
                              distn = []
                              mind = lamd_t[m, :, tau-1]
                              if tau == 1:
                                  F = length[k]
                              else:
                                  F = self.f
                              for p in range(F):                                  
                                  ploc = np.array(np.unravel_index(int(mind[p]), (self.size, self.size)))
                                  distn.append(np.linalg.norm(ploc - jloc))
                              dist += min(distn)  
                          
                          d[j] = np.abs(t_ind[j]) + (self.beta * (1/dist)) 
                      index = np.argmax(d)
                      ind.add(index) 
                      
                      if f == 0:
                          ind.add(self.image_size)                          
                      indt = list(ind)                            
                      ATA = 0
                      B = 0
                      for ch in range(self.ch):
                          tt = M2[:, k, ch]                          
                          ATA += np.dot(M1[:, indt, ch].T, M1[:, indt, ch])              
                          B += np.dot(M1[:, indt, ch].T, tt[:, np.newaxis])
                      A = np.linalg.inv(ATA+self.lamda*np.eye(len(ind))) # 
                      temp = np.dot(A,B).T
                      r = np.squeeze(tk - temp.dot(M1[:, indt, :])) 
                  return (indt, temp)

        data = Parallel(n_jobs=20)(delayed(compute_wi)(k) for k in range(self.image_size))
                
        for k in range(len(data)):
             ind = data[k]  
             print("***k", ind[0])                                
             W[k, ind[0]] = ind[1] 
            
 
        Wt = sparse.csr_matrix(W)         
        sparse.save_npz(str(self.weight_dir)+'lsrf_weights_N2HAS_f'+str(self.f)+'_img-size_'+str(self.size)+'.npz', Wt)

         
    
