#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:37:42 2020

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
from sklearn.metrics import r2_score as r_sq
import pandas as pd
from sklearn.preprocessing import normalize

import statsmodels.api as sm

Tensors, Eigenvectors, Eigenvalues, f_calc, f_eigval, g_calc, g_eigval, cn_calc, cn_eigval = loadData();
n = 500
r1 = 10 * np.ones(22)
phi1 = 0.35 * np.ones(22)
my_n = np.zeros(22)

num= 4

cn_calc_old = cn_calc[:]
cn_eigval_old = cn_eigval[:]

"""NEW DATA"""
Tensors = np.tile(Tensors,(num,1))
f_eigval = np.tile(f_eigval,(1, num)).squeeze()
g_eigval = np.tile(g_eigval,(1, num)).squeeze()
f_calc = np.tile(f_calc,(1, num)).squeeze()
g_calc = np.tile(g_calc,(1, num)).squeeze()

###  add 3 more values of r and phi  ###
r2 = np.concatenate((np.ones(22)*25, np.ones(22)*82, np.ones(22)*154)) 
phi2 = np.concatenate((np.ones(22)*0.22, np.ones(22)*0.4, np.ones(22)*0.58))
r = np.concatenate((r1,r2), axis=0)
phi = np.concatenate((phi1,phi2), axis=0)

###  calculate new values exactly and with eigenvalue functions  ###
cn_calc = 8/np.pi * phi * r * f_calc + 4 * phi * (g_calc + 1)
cn_eigval = 8/np.pi * phi * r * f_eigval + 4 * phi * (g_eigval + 1)


# assert (cn_calc_old == cn_calc[:22]).all()
# assert (cn_eigval_old == cn_eigval[:22]).all()

""" Input is vector with F and G functions calculated with eigenvalues + R and Phi values """
# X = np.stack((f_eigval, g_eigval, r, phi), axis=1) 

""" Input is vector with flattened matrix A """
# X = Tensors.reshape((22*num,9))
# X = np.concatenate((X, np.stack((r, phi), axis=1)), axis=1)

""" Input is vector with flattened matrix A + F and G functions calculated with eigenvalues """
X = Tensors.reshape((22*4,9))
X = np.concatenate((X, np.stack((f_eigval, g_eigval, r, phi), axis=1)), axis=1)

""" Input is vector with diagonal elements of matrix A + F and G functions calculated with eigenvalues """
# X = np.zeros((22*4,3))
# for i in range(22):
#     X[i,:] = np.diagonal(Tensors[i,:,:])
# X = np.concatenate((X, np.stack((f_eigval, g_eigval, r, phi), axis=1)), axis=1)

""" Target vector """
y = cn_calc

indices = np.arange(len(y))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                                    X, y, indices, test_size=0.33, random_state=25, stratify=X[:,12])

# p = np.random.permutation(len(y))
# X = X[p]
# y = y[p]
# indices = indices[p]

reg = LinearRegression(fit_intercept = True).fit(X, y)
#reg = sm.OLS(y_train, X_train).fit()


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

print("\nR2 for regression: {:.04f}".format(r_sq(overall_matrix[:,0], overall_matrix[:,1])))
print("R2 for eigenvalues: {:.04f}".format(r_sq(overall_matrix[:,0], overall_matrix[:,2])))


print("\n\n==================== ERRORS TEST ==================")
print("MSE for regression: {:.04f}".format(mse(test_matrix[:,0], test_matrix[:,1])))
print("MSE for eigenvalues: {:.04f}".format(mse(test_matrix[:,0], test_matrix[:,2])))

print("\nExplained variance for regression: {:.04f}".format(evs(test_matrix[:,0], test_matrix[:,1])))
print("Explained variance for eigenvalues: {:.04f}".format(evs(test_matrix[:,0], test_matrix[:,2])))


print("\n\n==================== COEFFICIENTS ==================")
#df = pd.DataFrame({'Parameter': ['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33', 'f_eigval', 'g_eigval'], 'Coefficients': getattr(reg, 'coef_')})
#df = pd.DataFrame({'Parameter': ['a11', 'a22', 'a33', 'f_eigval', 'g_eigval'], 'Coefficients': getattr(reg, 'coef_')})
#print(df)









# fig, ax = plt.subplots(figsize=(15,10))
# x = np.arange(22)
# #rects1 = ax.bar(x - width/2, overall_matrix[:,1], width, label='Regression')
# #rects2 = ax.bar(x + width/2, overall_matrix[:,0], width, label='Calculated')
# ax.bar(x-0.25, overall_matrix[:,0], width=0.25, align='center', label='Exact', color='#1b9e77', edgecolor='k')
# ax.bar(x, overall_matrix[:,1], width=0.25, align='center', label='Regression', color='#7570b3', edgecolor='k')
# ax.bar(x+0.25, overall_matrix[:,2], width=0.25, align='center', label='Eigenvalue Method', color="#d95f02", edgecolor='k')

# fig.suptitle("Exact vs Regression vs Eigenvalue Method", size = 20, x=0.5, y=0.95)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# plt.ylabel('Number of contact points', size = 16)

# # Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),
#           fancybox=True, shadow=True, ncol=5, prop={"size": 14})