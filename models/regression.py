# -*- coding: utf-8 -*-

import h5py
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Load prepared data
f = h5py.File('../dataset/dataset.h5', 'r')
X = f.get('X')[()]
y = f.get('y')[()]
X_pred = f.get('X_pred')[()]
y_pred = f.get('y_pred')[()]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Linear Regression Model
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)

# Lasso Regression    
lasso = linear_model.Lasso(alpha=1., max_iter=2000)
lasso.fit(X_train, y_train)

# Ridge Regression    
ridge = linear_model.Ridge(alpha=13.95, max_iter=2000)
ridge.fit(X_train, y_train)

# Scores
print('Ridge score is %f' % ridge.score(X_test, y_test))
print('Lasso score is %f' % lasso.score(X_test, y_test))
print('Linear Regression score is %f' % lr.score(X_test, y_test))

# Using different regression models, we got always 96% score

# Visualize the prediction
import matplotlib.pyplot as plt2
 
pred = lr.predict(X_pred)  
plt2.plot(pred, color='red', label='Prediction')
plt2.plot(y_pred, color='blue', label='Ground Truth')
plt2.legend(loc='upper left')
plt2.show()




