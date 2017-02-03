""" Glass identification exercise: https://www.kaggle.com/uciml/glass"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# for prettier plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 10)

#import dataset
glass = pd.read_csv('glass.csv')

# check correlations of unstandardized dataset
variable_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
corr = glass[variable_cols].corr()
cmap = sns.cubehelix_palette(start = 2, rot = 0, dark = 0, light = .95, as_cmap = True)
sns.heatmap(corr, cmap = cmap, annot = True, fmt = '.2f')
plt.title('Correlation matrix of the glass variables')
plt.show()

# Description of dataset
print glass.describe()

# Boxplot of variables
glass[variable_cols].boxplot()
plt.show()
variable_noSi = ['RI', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Ba', 'Fe']
glass[variable_noSi].boxplot()
plt.show()

# Convert to numpy arrays
vars = glass.as_matrix(columns=variable_cols)
resp  = glass.as_matrix(columns = ['Type']).ravel()

# Standardise & plot
sc = StandardScaler()
vars_std = sc.fit_transform(vars)

corr_std = np.corrcoef(vars_std, rowvar = 0)

f, (ax1, ax2) = plt.subplots(2)
sns.heatmap(corr, cmap = cmap, annot = True, fmt = '.2f', ax = ax1)
ax1.set_title('Correlation matrix of the glass variables')
sns.heatmap(corr_std, cmap = cmap, annot = True, fmt = '.2f', ax = ax2)
ax2.set_title('Correlation matrix of the standardized glass variables')
ax2.set_xticklabels(variable_cols)
ax2.set_yticklabels(variable_cols[::-1])
plt.show()
### => Standardization does not change the covariance matrix ... duuuh!

## CLASSIFICATION
# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(vars_std, resp, train_size=0.70)

# SVC
svc = SVC(probability=True)
param = {
	"C" : [0.1,1,10,100],
	"kernel" : ['rbf', 'poly'],
	"degree" : [2, 4, 6],
	"gamma" : ['auto', 1, 5, 10]
	}
svc_gs = GridSearchCV(estimator = svc, param_grid=param, cv=5, refit=True, scoring='accuracy')
svc_gs.fit(x_train, y_train)
svc = svc_gs.best_estimator_
svc.fit(x_train, y_train)

