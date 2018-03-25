# -*- coding: utf-8 -*-

import h5py
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Load prepared data
f = h5py.File('../dataset/dataset.h5', 'r')
X = f.get('X')[()]
y = f.get('y')[()]
X_pred = f.get('X_pred')[()]
y_pred = f.get('y_pred')[()]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# KNN Model
knn = KNeighborsRegressor(algorithm='brute')
knn.fit(X_train, y_train)

# Score
print('KNN score is %f' % knn.score(X_test, y_test))

# Visualize the prediction
import matplotlib.pyplot as plt2

pred = knn.predict(X_pred)
    
plt2.plot(pred, color='red', label='Prediction')
plt2.plot(y_pred, color='blue', label='Ground Truth')
plt2.legend(loc='upper left')
plt2.show()

