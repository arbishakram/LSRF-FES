#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:46:46 2022

@author: arbish
"""

import numpy as np
import cv2
from scipy import sparse
from f_load_data import *
from lsrf import *
from omp import *
from f_create_mask import create_mask
import json

class omp_solver:
     """Solver for training and testing."""
     
     def __init__(self,args):
        self.size = args.image_size
        self.image_size = self.size * self.size 
        self.size_n = self.image_size+1 
        self.ch = args.input_ch
        self.lamda = args.lamda       
        self.beta = args.beta
        self.weight_dir = args.weights_dir
        self.train_dataset_dir = args.train_dataset_dir
        self.val_dataset_dir = args.val_dataset_dir        
        self.test_dataset_dir = args.test_dataset_dir
        self.names = []       
        self.f = args.f
        self.model_name = args.model
       
     def train(self, args):       
         path = str(self.weight_dir)+'arguments_history.txt'
         with open(path, 'w') as f:
              json.dump(args.__dict__, f, indent=2)           
         print("loading input and target images...")
         train_imagesA, train_imagesB = load_data(self, args, args.train_dataset_dir, 'train')
         print(train_imagesA.shape, train_imagesB.shape)
        
         
         local_neighbors, Mk = create_mask(args)
         
         for ch in range(1):
             print("form design and repsonse matrices.... ")
             train_M1, train_M2 = create_design_response_matrices(self, train_imagesA, train_imagesB)   
             print(train_M1.shape, train_M2.shape)           

             print("learn MR weights....")
             
             if self.model_name == 'LSRF':
                 W =  omp_with_locality_constraint(self, local_neighbors, Mk, train_M1, train_M2)
             elif self.model_name == 'OMP':
                 W =  omp(self, train_M1, train_M2)  
             else:
                 print("Please enter the correct method name from the following option: \n omp \n lsrf")
             print("saved weight matrix...")
             print("*"*40)
             print("Done")
             
    
     def test(self, args):
             print("loading in the wild images...")
             _, self.n = load_data(self, args, args.test_dataset_dir, 'test_inthewild')
             
             for p in range(self.n):
                 xn = np.zeros((self.size,self.size,self.ch))   
                 ynt = np.zeros((self.size,self.size,self.ch))
                 Wt = sparse.load_npz(str(args.weights_dir)+'omp_weights_N2HAS_f'+str(self.f)+'_img-size_'+str(self.size)+'.npz') 
                 Wt = Wt.todense()
                 for ch in range(self.ch):       
                        x = load_x(self, p,ch)                          
                        if len(x)  != 0:  
                           W = Wt[:, :self.image_size]
                           b = Wt[:, -1]
                           a = np.dot(W,x[:self.image_size]).T
                           yn = a + b
                           xnt = x[0:self.image_size]
                           xn[:,:,ch] = np.reshape(xnt,(self.size,self.size))
                           ynt[:,:,ch] = np.reshape(yn,(self.size,self.size))    
                           
                 ynt = cv2.normalize(ynt, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                 xn = cv2.normalize(xn, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)                  
                 ftest = np.concatenate((xn,ynt), axis=1) 
                 cv2.imwrite(str(args.results_dir)+str(self.names[p])+'_y_1_test.png',ynt)
                 print(p,'Saved input and output images into {}'.format(args.results_dir))

                 
