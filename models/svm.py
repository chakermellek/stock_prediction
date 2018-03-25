# -*- coding: utf-8 -*-

import h5py
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Load prepared data
f = h5py.File('../dataset/dataset.h5', 'r')
X = f.get('X')[()]
y = f.get('y')[()]
X_truth = f.get('X_pred')[()]
y_truth = f.get('y_pred')[()]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Linear support vector regression model
svr_lin = SVR(kernel= 'linear', C= 1e3)
svr_lin.fit(X_train, y_train)

# Polynomial support vector regression model
svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
svr_poly.fit(X_train, y_train)

# Radial basis function support vector regression model
svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
svr_rbf.fit(X_train, y_train)

# Scores
print('Linear SVR score is %f' % svr_lin.score(X_test, y_test))
print('Polynomial SVR is %f' % svr_poly.score(X_test, y_test))
print('Radial basis function SVR score is %f' % svr_rbf.score(X_test, y_test))

# The best score is 98% using Radial basis function SVR

# Visualize the prediction
import matplotlib.pyplot as plt2

pred = svr_rbf.predict(X_truth)
    
plt2.plot(pred, color='red', label='Prediction')
plt2.plot(y_truth, color='blue', label='Ground Truth')
plt2.legend(loc='upper left')
plt2.show()

