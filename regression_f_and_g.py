#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:46:27 2020

@author: jonona
"""


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
from sklearn.metrics import r2_score as r_sq
import pandas as pd
from sklearn.preprocessing import normalize

import statsmodels.api as sm


Tensors, Eigenvectors, Eigenvalues, f_calc, f_eigval, g_calc, g_eigval, cn_calc, cn_eigval, angle_calc, angle_eigval = loadData(angle_switch=True);
n = 500
r = 10 * np.ones(22)
phi = 0.35 * np.ones(22)


"""NEW DATA"""


# assert (cn_calc_old == cn_calc[:22]).all()
# assert (cn_eigval_old == cn_eigval[:22]).all()

""" Input is vector with flattened matrix A """
X = Tensors.reshape((22,9))

""" Input is vector with diagonal elements of matrix A """
# X = np.zeros((22,3))
# for i in range(22):
#     X[i,:] = np.diagonal(Tensors[i,:,:])

""" Target vector """
y = np.stack((f_calc, g_calc)).transpose()

indices = np.arange(len(y))


reg = LinearRegression(fit_intercept = True).fit(X, y)
#reg = sm.OLS(y_train, X_train).fit()


# test_matrix = np.stack((y_test,reg.predict(X_test),cn_eigval[idx_test]), axis=1)
# train_matrix = np.stack((y_train,reg.predict(X_train),cn_eigval[idx_train]), axis=1)
overall_matrix_f = np.stack((y[:,0],reg.predict(X)[:,0], f_eigval), axis=1)

overall_matrix_g = np.stack((y[:,1],reg.predict(X)[:,1], g_eigval), axis=1)


# # # adding indices to sort examples as in the presentation
# overall_matrix_f = sortieren(overall_matrix_f, indices)
# overall_matrix_g = sortieren(overall_matrix_g, indices)

print("\n==================== RESULTS F ==================")
print("Ground truth    Regression   Eigenvalue")
print(overall_matrix_f)

print("\n==================== RESULTS G ==================")
print("Ground truth    Regression   Eigenvalue")
print(overall_matrix_g)


print("\n\n==================== ERRORS OVERALL F ==================")
print("MSE for regression: {:.04f}".format(mse(overall_matrix_f[:,0], overall_matrix_f[:,1])))
print("MSE for eigenvalues: {:.04f}".format(mse(overall_matrix_f[:,0], overall_matrix_f[:,2])))

print("\nR2 for regression: {:.04f}".format(r_sq(overall_matrix_f[:,0], overall_matrix_f[:,1])))
print("R2 for eigenvalues: {:.04f}".format(r_sq(overall_matrix_f[:,0], overall_matrix_f[:,2])))


print("\n\n==================== ERRORS OVERALL G ==================")
print("MSE for regression: {:.04f}".format(mse(overall_matrix_g[:,0], overall_matrix_g[:,1])))
print("MSE for eigenvalues: {:.04f}".format(mse(overall_matrix_g[:,0], overall_matrix_g[:,2])))

print("\nR2 for regression: {:.04f}".format(r_sq(overall_matrix_g[:,0], overall_matrix_g[:,1])))
print("R2 for eigenvalues: {:.04f}".format(r_sq(overall_matrix_g[:,0], overall_matrix_g[:,2])))


# print("\n\n==================== COEFFICIENTS ==================")
# #df = pd.DataFrame({'Parameter': ['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33', 'f_eigval', 'g_eigval'], 'Coefficients': getattr(reg, 'coef_')})
# #df = pd.DataFrame({'Parameter': ['a11', 'a22', 'a33', 'f_eigval', 'g_eigval'], 'Coefficients': getattr(reg, 'coef_')})
# #print(df)


###  CALCULATING NC  ###
cn_new = 8/np.pi * phi * r * overall_matrix_f[:,1] + 4 * phi * (overall_matrix_g[:,1] + 1)

print("\n\n==================== ERRORS OVERALL NUMBER OF CONTACTS ==================")
print("MSE for regression: {:.04f}".format(mse(cn_new, cn_calc)))
print("MSE for eigenvalues: {:.04f}".format(mse(cn_eigval, cn_calc)))

print("\nR2 for regression: {:.04f}".format(r_sq(cn_new, cn_calc)))
print("R2 for eigenvalues: {:.04f}".format(r_sq(cn_eigval, cn_calc)))


print("\n\n==================== COEFFICIENTS ==================")
df = pd.DataFrame({'Parameter': ['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33'], 'Coefficients for f': getattr(reg, 'coef_')[0], 'Coefficients for g': getattr(reg, 'coef_')[1]})
print(df)



fig, ax = plt.subplots(figsize=(15,10))
x = np.arange(22)
ax.bar(x-0.25, overall_matrix_f[:,0], width=0.25, align='center', label='Exact', color='#1b9e77', edgecolor='k')
ax.bar(x, overall_matrix_f[:,1], width=0.25, align='center', label='Regression', color='#7570b3', edgecolor='k')
ax.bar(x+0.25, overall_matrix_f[:,2], width=0.25, align='center', label='Eigenvalue Method', color="#d95f02", edgecolor='k')

fig.suptitle("FUNCTION F", size = 20, x=0.5, y=0.95)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width, box.height * 0.9])
plt.ylabel('f', size = 16)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=5, prop={"size": 14})
plt.ylim([0,1.1])

fig, ax = plt.subplots(figsize=(15,10))
ax.bar(x-0.25, overall_matrix_g[:,0], width=0.25, align='center', label='Exact', color='#1b9e77', edgecolor='k')
ax.bar(x, overall_matrix_g[:,1], width=0.25, align='center', label='Regression', color='#7570b3', edgecolor='k')
ax.bar(x+0.25, overall_matrix_g[:,2], width=0.25, align='center', label='Eigenvalue Method', color="#d95f02", edgecolor='k')

fig.suptitle("FUNCTION G", size = 20, x=0.5, y=0.95)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width, box.height * 0.9])
plt.ylabel('g', size = 16)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=5, prop={"size": 14})
plt.ylim([0,1.1])