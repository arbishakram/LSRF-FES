#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:46:46 2022

@author: arbish
"""

import argparse
import os
from solver import *
import time

start = time.time()
exp_name = 'Exp_N2HAS_128_omp_k_1/'
dataset_path = './dataset/N2HAS/'
inthewild_path = ''

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_dataset_dir', dest='train_dataset_dir', default=dataset_path+'train/', help='path of the train dataset')
parser.add_argument('--val_dataset_dir', dest='val_dataset_dir', default=dataset_path+'val/', help='path of the validation dataset')
parser.add_argument('--test_dataset_dir', dest='test_dataset_dir', default=inthewild_path+'imgs/', help='path of the inthewild test dataset')
parser.add_argument('--image_size', dest='image_size', type=int, default=128, help='size of image')
parser.add_argument('--input_ch', dest='input_ch', type=int, default=3, help='# of input image channels')
parser.add_argument('--lamda', dest='lamda', type=float, default=0.4, help='lambda value')
parser.add_argument('--beta', dest='beta', type=float, default=60, help='beta value')
parser.add_argument('--f', dest='f', type=int, default=5, help='number of nonzero coefficients')
parser.add_argument('--model', dest='model', default='OMP', help='LSRF, OMP')
parser.add_argument('--mode', dest='mode', default='test_inthewild', help='train, test_inthewild')
parser.add_argument('--weights_dir', dest='weights_dir', default=exp_name+'weights/', help='weights are saved here')
parser.add_argument('--results_dir', dest='results_dir', default=exp_name+'omp_5/', help='inthewild results are saved here')
args = parser.parse_args()




def main():   
    # Create directories if not exist.
    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
  
    # Solver for training and testing.
    model = omp_solver(args)
    model.train(args) if args.mode == 'train' \
        else model.test(args) 

main()
end = time.time()
temp = end - start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('%d:%d:%d' %(hours,minutes,seconds))
