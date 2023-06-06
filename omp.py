# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:37:43 2023

@author: arbish
"""

import numpy as np
from scipy import sparse
from sklearn.preprocessing.data import normalize as f_normalize
from joblib import Parallel, delayed

def omp(self, M1, M2):    
     W = np.zeros((self.image_size, self.size_n))    
     Xtt = M1[:, :self.image_size, :]     
     le = Xtt.shape[0]
     norm_Xtt = np.zeros((le, self.image_size, 3))     
     X_offset = np.average(Xtt, axis=0)     
     Xtt = Xtt - X_offset
     
     # normalize each feature
     for ch in range(self.ch):
         tXtt, X_scale = f_normalize(Xtt[:,:,ch], axis=0, copy=False, return_norm=True)   
         norm_Xtt[:,:,ch] = tXtt 


     W = np.zeros((self.image_size, self.size_n))         
     def compute_wi(k):
     # for k in range(self.image_size):  
         print(k)
         tk = M2[:, k, :]
         r = M2[:, k, :]    
         ind = set()
         omega = []
         for f in range(self.f):     
             t_ind = 0
             for ch in range(3):
                 t_ind +=  np.dot(norm_Xtt[:, :, ch].T, r[:, ch])   
             t_ind = np.abs(t_ind)
             for p in range(len(omega)):
                 index = omega[p]
                 t_ind[index] = -100
             ind.add(np.argmax(t_ind))  
             omega.append(np.argmax(t_ind))
             if f == 0:
                 ind.add(self.image_size)  
             indt = list(ind)
             ATA = 0
             B = 0
             for ch in range(self.ch):
                 tt = M2[:, k, ch]
                 ATA += np.dot(M1[:, indt, ch].T, M1[:, indt, ch])              
                 B += np.dot(M1[:, indt, ch].T, tt[:, np.newaxis])
             A = np.linalg.inv(ATA+self.lamda*np.eye(len(ind)))
             temp = np.dot(A,B).T
             r = np.squeeze(tk - temp.dot(M1[:, indt, :]))
         return (indt, temp)
         
     data = Parallel(n_jobs=18)(delayed(compute_wi)(k) for k in range(self.image_size))
     for ii in range(len(data)):
              ind = data[ii]
              print("***k", ind[0])
              W[ii, ind[0]] = ind[1]   

     W = sparse.csr_matrix(W)
     sparse.save_npz(str(self.weight_dir)+'omp_weights_dec-2021_'+str(self.f)+'.npz', W)
     return W    

