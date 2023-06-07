#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:46:46 2022

@author: arbish
"""

import os
import glob
import numpy as np
import cv2
from f_normalize_images import normalize_img
import natsort

def load_data(self, args, in_path, mod):

             def get_img(img):
                if self.ch == 1:
                    image = cv2.imread(img,0)
                    image = cv2.resize(image, (self.size, self.size))
                    image = normalize_img(image)                       
                    img = np.zeros(self.image_size*self.ch)                
                    img[0:self.image_size] = np.reshape(image,[self.image_size])
                else:
                    image = cv2.imread(img)
                    image = cv2.resize(image, (self.size, self.size))
                    image = normalize_img(image)   
                    img = np.zeros(self.image_size*self.ch)                
                    img[0:self.image_size] = np.reshape(image[:,:,0],[self.image_size])
                    img[self.image_size:self.image_size*2] = np.reshape(image[:,:,1],[self.image_size])
                    img[self.image_size*2:self.image_size*3] = np.reshape(image[:,:,2],[self.image_size])               
                return img
         
            
             if args.mode == 'test_inthewild':                 
                 in_path = args.test_dataset_dir
                 imgA=glob.glob(os.path.join(in_path, '*.jpg'))
                 imgA=natsort.natsorted(imgA)
                 print('Number of images in '+str(args.mode)+':',len(imgA))                 
                 n = len(imgA)                 
                 self.imagesA = np.zeros((n,self.image_size*self.ch))                 
                 count = 0
                 for img in imgA:   
                    img_name = os.path.basename(img)
                    filename, ext = os.path.splitext(img_name)
                    self.names.append(filename)
                    img = get_img(img)
                    self.imagesA[count] = np.array(img)
                    count = count + 1   
                 return self.imagesA, n
                    
             else:                 
                 imgA=glob.glob(os.path.join(in_path, str(mod)+'A/*.png'))
                 imgA=natsort.natsorted(imgA)
                 print('Number of images in '+str(mod)+'A:',len(imgA))                    
                 imgB=glob.glob(os.path.join(in_path, str(mod)+'B/*.png'))
                 imgB=natsort.natsorted(imgB)
                 print('Number of images in '+str(mod)+'B:',len(imgB))                 
                 # print(imgB, imgA)
                 self.n = len(imgA)                 
                 self.imagesA = np.zeros((self.n,self.image_size*self.ch))
                 self.imagesB = np.zeros((self.n,self.image_size*self.ch))                
                 cc = 0
                 for img in imgA:   
                    img_name = os.path.basename(img)
                    filename, ext = os.path.splitext(img_name)
#                    num = filename.split('_')[1]
                    self.names.append(filename)
                    img = get_img(img)
#                    print(filename)
#                    self.imagesA[int(num)-1] = np.array(img)
                    self.imagesA[cc] = np.array(img)
                    cc = cc+1                
              
                 cc = 0
                 for img in imgB:
                    img_name = os.path.basename(img)
                    filename, ext = os.path.splitext(img_name)
#                    num = filename.split('_')[1]
                    img = get_img(img)
#                    self.imagesB[int(num)-1] = np.array(img)
                    self.imagesB[cc] = np.array(img)
                    cc = cc+1
                    
             return self.imagesA, self.imagesB
         
#            
            
def load_x_t(self, p,ch,imagesA,imagesB):           
            if ch==0:        
                x = imagesA[p,0:self.image_size] 
                t = imagesB[p,0:self.image_size]
            if ch==1:
                x = imagesA[p,self.image_size:self.image_size*2] 
                t = imagesB[p,self.image_size:self.image_size*2]
                
            if ch==2:
                x = imagesA[p,self.image_size*2:self.image_size*3] 
                t = imagesB[p,self.image_size*2:self.image_size*3]
            xt = np.ones(self.size_n)
            xt[0:self.image_size] = x
            return xt,t
        
            
        
def load_x(self, p, ch):           
            if ch==0:        
                x = self.imagesA[p,0:self.image_size] 
            if ch==1:
                x = self.imagesA[p,self.image_size:self.image_size*2]                    
            if ch==2:
                x = self.imagesA[p,self.image_size*2:self.image_size*3] 
            xt = np.ones(self.size_n)   
            xt[0:self.image_size] = x              
            return xt

                

def create_design_response_matrices(self, imagesA, imagesB):
        n = imagesA.shape[0]
    
        M1 = np.ones((n, self.size_n,3))
        M2 = np.zeros((n, self.image_size,3))
        r=0
        
        for ch in range(3):
            r = 0
            for p in range(n):  
                            x, t = load_x_t(self, p, ch, imagesA, imagesB) 
                            if len(x) or len(t) != 0:
                                x = x.T                 
                                tt = np.reshape(t,[1,self.image_size])  
                                M1[r,:,ch] = x
                                M2[r,:,ch] = tt
                                r = r+1
        M1 = np.delete(M1,np.where(~M2.any(axis=1))[0], axis=0)
        M2 = np.delete(M2,np.where(~M2.any(axis=1))[0], axis=0)    
        print(M1.shape, M2.shape)
        return M1, M2
    
