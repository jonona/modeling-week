#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:25:27 2020

@author: jonona
"""


import numpy as np
from loadData import *
import matplotlib.pyplot as plt
#from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2
import pandas as pd

import statsmodels.api as sm

Tensors, Eigenvectors, Eigenvalues, f_calc, f_eigval, g_calc, g_eigval, cn_calc, cn_eigval = loadData();
n = 500
r = 10 * np.ones(22)
phi = 0.35 * np.ones(22)
my_n = np.zeros(22)

""" Input is vector with F and G functions calculated with eigenvalues """
# X = np.stack((f_eigval, g_eigval), axis=1) 

""" Input is vector with F and G functions calculated with eigenvalues + R and Phi values """
# X = np.stack((f_eigval, g_eigval, r, phi), axis=1) 

""" Input is vector with flattened matrix A """
# X = Tensors.reshape((22,9))

""" Input is vector with flattened matrix A + F and G functions calculated with eigenvalues """
X = Tensors.reshape((22,9))
X = np.concatenate((X, np.stack((f_eigval, g_eigval), axis=1)), axis=1)

""" Input is vector with diagonal elements of matrix A + F and G functions calculated with eigenvalues """
# X = np.zeros((22,3))
# for i in range(22):
#     X[i,:] = np.diagonal(Tensors[i,:,:])
# X = np.concatenate((X, np.stack((f_eigval, g_eigval), axis=1)), axis=1)

""" Target vector """
y = cn_calc

indices = np.arange(22)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                                    X, y, indices, test_size=0.33, random_state=25)

reg = LinearRegression().fit(X, y)
#reg = sm.OLS(y_train, X_train).fit()

indices = np.concatenate((idx_test, idx_train))

test_matrix = np.stack((y_test,reg.predict(X_test),cn_eigval[idx_test]), axis=1)
train_matrix = np.stack((y_train,reg.predict(X_train),cn_eigval[idx_train]), axis=1)
overall_matrix = np.concatenate((test_matrix, train_matrix), axis=0)

# adding indices to sort examples as in the presentation
overall_matrix = np.concatenate((overall_matrix,np.expand_dims(indices, axis=1)), axis=1)
overall_matrix = overall_matrix[overall_matrix[:, 3].argsort()]
overall_matrix = overall_matrix[:,:3]

print("\n==================== TESTING ==================")
print("Ground truth    Regression   Eigenvalue")
print(test_matrix)

print("\n\n==================== FITTED ==================")
print("Ground truth   Regression   Eigenvalue")
print(train_matrix)

print("\n\n==================== ERRORS OVERALL ==================")
print("MSE for regression: {:.04f}".format(mse(overall_matrix[:,0], overall_matrix[:,1])))
print("MSE for eigenvalues: {:.04f}".format(mse(overall_matrix[:,0], overall_matrix[:,2])))

print("\nExplained variance for regression: {:.04f}".format(evs(overall_matrix[:,0], overall_matrix[:,1])))
print("Explained variance for eigenvalues: {:.04f}".format(evs(overall_matrix[:,0], overall_matrix[:,2])))

print("\nR2 for regression: {:.04f}".format(r2(overall_matrix[:,0], overall_matrix[:,1])))
print("R2 for eigenvalues: {:.04f}".format(r2(overall_matrix[:,0], overall_matrix[:,2])))


print("\n\n==================== ERRORS TEST ==================")
print("MSE for regression: {:.04f}".format(mse(test_matrix[:,0], test_matrix[:,1])))
print("MSE for eigenvalues: {:.04f}".format(mse(test_matrix[:,0], test_matrix[:,2])))

print("\nExplained variance for regression: {:.04f}".format(evs(test_matrix[:,0], test_matrix[:,1])))
print("Explained variance for eigenvalues: {:.04f}".format(evs(test_matrix[:,0], test_matrix[:,2])))


print("\n\n==================== COEFFICIENTS ==================")
# df = pd.DataFrame({'Parameter': ['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33', 'f_eigval', 'g_eigval'], 'Coefficients': getattr(reg, 'coef_')})
# #df = pd.DataFrame({'Parameter': ['a11', 'a22', 'a33', 'f_eigval', 'g_eigval'], 'Coefficients': getattr(reg, 'coef_')})
# print(df)









fig, ax = plt.subplots(figsize=(15,10))
x = np.arange(22)
ax.bar(x-0.25, overall_matrix[:,0], width=0.25, align='center', label='Exact', color='#1b9e77', edgecolor='k')
ax.bar(x, overall_matrix[:,1], width=0.25, align='center', label='Regression', color='#7570b3', edgecolor='k')
ax.bar(x+0.25, overall_matrix[:,2], width=0.25, align='center', label='Eigenvalue Method', color="#d95f02", edgecolor='k')

fig.suptitle("Exact vs Regression vs Eigenvalue Method", size = 20, x=0.5, y=0.95)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
plt.ylabel('Number of contact points', size = 16)

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=5, prop={"size": 14})


# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, overall_matrix[:,2], width, label='Eigenvalue')
# rects2 = ax.bar(x + width/2, overall_matrix[:,0], width, label='Calculated')
# fig.suptitle("Eigenvalue vs Calculated")
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# # Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
#           fancybox=True, shadow=True, ncol=5)
