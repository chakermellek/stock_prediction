# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import h5py
from sklearn.utils import shuffle
import sklearn.preprocessing as pp

# Import data
df = pd.read_csv('DataBase-2000.csv', na_values=['#N/A'], delimiter=';')
                                                                                                    
# Remove Whitespaces from column names
df.columns = df.columns.str.strip()

# Drop useless variables
df = df.drop(['Date'], axis=1)
df = df.drop(['GSEACII Index'], axis=1)
df = df.drop(['GSUSCII Index'], axis=1)
df = df.drop(['GSJPCII Index'], axis=1)

df = df.drop(['EUR003M Index'], axis=1)
df = df.drop(['US0003M Index'], axis=1)
df = df.drop(['BP0003M Index'], axis=1)
df = df.drop(['JY0003 Index'], axis=1)

# Visualize the dataset
# =============================================================================
# import matplotlib.pyplot as plt
# 
# plt.scatter(df['DAX Index'],df['CAC Index'],s=1)
# 
# plt.title('Nuage de points avec Matplotlib')
# plt.xlabel('DAX Index')
# plt.ylabel('CAC Index')
# plt.savefig('ScatterPlot.png')
# plt.show()
# =============================================================================

# Convert string values with comma to float
df = df.apply(lambda x: x.str.replace(',','.')).astype('float')

# fill missing values with mean column values
#df.fillna(df.mean(), inplace=True)

# Or Drop all observations containing NaN Values
df.dropna(how='any') 

# Visualize all variables evolution in time 
import matplotlib.pyplot as plt2
plt2.figure(figsize=(10,10))
plt2.plot(df['CAC Index'], color='tab:brown', label='CAC Index', marker="o",  markersize=1)
plt2.plot(df['SPX Index'], color='b', label='SPX Index', marker="o",  markersize=1)
plt2.plot(df['UKX Index'], color='g', label='UKX Index', marker="o",  markersize=1)
plt2.plot(df['DAX Index'], color='r', label='DAX Index', marker="o",  markersize=1)
plt2.plot(df['NKY Index'], color='c', label='NKY Index', marker="o",  markersize=1)
plt2.plot(df['INDU Index'], color='m', label='INDU Index', marker="o",  markersize=1)
plt2.plot(df['EUR curncy'], color='y', label='EUR curncy', marker="o",  markersize=1)
plt2.plot(df['EURJPY curncy'], color='k', label='EURJPY curncy', marker="o",  markersize=1)
plt2.plot(df['EURGBP curncy'], color='w', label='EURGBP curncy', marker="o",  markersize=1)
plt2.plot(df['GOLDS Comdty'], color='tab:pink', label='GOLDS Comdty', marker="o",  markersize=1)
plt2.plot(df['CL1 COMB Comdty'], color='tab:purple', label='CL1 COMB Comdty', marker="o",  markersize=1)

plt2.legend(loc='upper right')
plt2.savefig('Correlation.png')
plt2.show()

# Seperate attributes from targets
y = df['CAC Index']
df = df.drop('CAC Index', axis=1)
X = df.as_matrix()

# Normalize and center numeric values
scaler = pp.StandardScaler()
scaler.fit(X)
X = scaler.transform(X) # X normalized

# Extract data on which we will do prediction (data from from 13/10/2017 to 12/03/2018)
y_pred = y.loc[0:106]
X_pred = X[0:106, :]

#Shuffle data
X, y = shuffle(X, y)


# Save data in a h5 file
dataset = {'X': X, 'y': y, 'X_pred': X_pred, 'y_pred': y_pred}
f = h5py.File('dataset.h5')
for key in dataset:
    f.create_dataset(key, data=dataset[key])
f.close()

print('Done.')
    
                     