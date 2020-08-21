#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 18:25:25 2020

@author: jonona
"""


import numpy as np
from loadData import *
import matplotlib.pyplot as plt
from plotting import *


Tensors, Eigenvectors, Eigenvalues, f_calc, f_eigval, g_calc, g_eigval, cn_calc, cn_eigval = loadData();
n = 500
r = 10
phi = 0.35
my_n = np.zeros(22)

for i in range(22):
    print("Number {}".format(i+1))
    print("\nMatrix ")
    a = Tensors[i]
    #a = np.multiply(Tensors[i],np.transpose(Tensors[i]))
    print(a)
    values, vectors = np.linalg.eig(a)
    print("\nEigenvalues ")
    print(values)
    print("Eigenvectors ")
    print(vectors)

    f_num = 3*np.pi/4*(values[1]*values[2] + values[1]*values[0] + values[0]*values[2])
    print("\nF numerical = {:.04f}".format(f_num))
    print("F initial numerical = {:.04f}".format(f_eigval[i]))
    print("F calculated = {:.04f}\n".format(f_calc[i]))
    
    g_num = (values[1]*values[1] + values[0]*values[0] + values[2]*values[2]) #added a factor 2pi/5
    print("G numerical = {:.04f}".format(g_num))
    print("G initial numerical = {:.04f}".format(g_eigval[i]))
    print("G calculated = {:.04f}\n".format(g_calc[i]))
    
    
    n_c = 8/np.pi*phi*r*f_num + 4*phi*(g_num+1)
    my_n[i] = n_c
    print("N numerical = {:.04f}".format(n_c))
    print("N initial numerical = {:.04f}".format(cn_eigval[i]))
    print("N calculated = {:.04f}".format(cn_calc[i]))
    print("My difference = {:.04f}".format(cn_calc[i]-n_c))
    print("Initial difference = {:.04f}".format(cn_calc[i]-cn_eigval[i]))
    print("==================================\n")
    
    ellipsoid(i,a)
    
print("Overall difference mine = {:.04f}".format(np.linalg.norm(my_n-cn_calc)))
print("Overall difference initial = {:.04f}".format(np.linalg.norm(cn_eigval-cn_calc)))
    
fig, ax = plt.subplots()
x = np.arange(22)
width = 0.35
rects1 = ax.bar(x - width/2, my_n, width, label='Mine')
rects2 = ax.bar(x + width/2, cn_calc, width, label='Calculated')
ax.legend()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, cn_eigval, width, label='Eigenvalue')
rects2 = ax.bar(x + width/2, cn_calc, width, label='Calculated')
ax.legend()