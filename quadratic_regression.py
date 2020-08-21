

from loadData import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


Tensors, Eigenvectors, Eigenvalues, f_calc, f_eigval, g_calc, g_eigval, cn_calc, cn_eigval = loadData()

n = 500
r = 10 * np.ones(22)
phi = 0.35 * np.ones(22)

""" Input is vector with F and G functions calculated with eigenvalues """
#x = np.stack((f_eigval, g_eigval), axis=1)


""" Input is vector with F and G functions calculated with eigenvalues + R and Phi values """
#x = np.stack((f_eigval, g_eigval, r, phi), axis=1)

""" Input is vector with flattened matrix A """
x = Tensors.reshape((22,9))

""" Input is vector with flattened matrix A + F and G functions calculated with eigenvalues """
#x = Tensors.reshape((22,9))
#x = np.concatenate((x, np.stack((f_eigval, g_eigval), axis=1)), axis=1)
#print(x)


""" Input is vector with flattenediagonal elements of matrix A + F and G functions calculated with eigenvalues """
#x = np.zeros((22,3))
#for i in range(22):
#    x[i,:] = np.diagonal(Tensors[i,:,:])
#x = np.concatenate((x, np.stack((f_eigval, g_eigval), axis=1)), axis=1)

"""Targeted Vector"""

y = cn_calc



polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)
for w in range(len(x_poly)):
    print(x_poly[w])


model = LinearRegression().fit(x_poly,y)
y_poly_pred = model.predict(x_poly)
#print(y_poly_pred)


#rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
#r2 = r2_score(y,y_poly_pred)
#mse = (rmse)*(rmse)
#print('Mean Square Error is:', mse)
#print('R-squared of the model is:', r2)


#fig, ax = plt.subplots(figsize=(15,10))
#x = np.arange(22)
#ax.bar(x-0.25, cn_calc, width=0.25, align='center', label='Exact', color='#1b9e77', edgecolor='k')
#ax.bar(x, y_poly_pred, width=0.25, align='center', label='Quadratic Polynomial', color='#7570b3', edgecolor='k')
#ax.bar(x+0.25, cn_eigval, width=0.25, align='center', label='Eigenvalue Method', color="#d95f02", edgecolor='k')

#fig.suptitle("Exact vs Quadratic Polynomial  vs Eigenvalue Method", size = 20, x=0.5, y=0.95)
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])
#plt.ylabel('Number of contact points', size = 16)

#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),
#          fancybox=True, shadow=True, ncol=5, prop={"size": 14})
#plt.show()



