#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:48:13 2020

@author: jonona
"""
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ellipsoid(number,A):
    #N = 20; 
    U, s, rotation = np.linalg.svd(A); 
    center = [0,0,0]
    
    #---------------------------------- 
    # generate the ellipsoid at (0,0,0) 
    #---------------------------------- 
    radii = 1.0/np.sqrt(s) 
     
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='k', alpha=0.2)
    fig.suptitle('Example number {}'.format(number), fontsize=14)
    plt.show()
    plt.close(fig)
    del fig