""" Glass identification exercise: https://www.kaggle.com/uciml/glass"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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
resp_names = map(str, np.unique(resp).tolist())

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

####
#SVC
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

print "SVC classifier\nAccuracy on training set: %.2f" % svc.score(x_train, y_train)
print "The accuracy of  testing set: %.2f" % svc.score(x_test, y_test)

predict_x_test = svc.predict(x_test)
print "Accuracy score: %.2f" % accuracy_score(y_test, predict_x_test)
print "Precision score: %.2f" % precision_score(y_test, predict_x_test, average='weighted')
print  "Recall score: %.2f" % recall_score(y_test, predict_x_test, average='weighted')
print "F1 score: %.2f" % f1_score(y_test, predict_x_test, average='weighted')

# check out the confusion matrix & classification report
print "\nConfusion matrix: \n", confusion_matrix(y_test, predict_x_test)
print "\nClassification report: \n", classification_report(y_test, predict_x_test, target_names=resp_names)
print "-"*55

# SVC not running too great - success rate of only ~66%

#############
# Reduce number of features  [feature importance] and rerun SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# Feature importance
etc = ExtraTreesClassifier()
etc = etc.fit(vars, resp)
print "Feature importance: \n", etc.feature_importances_
transform_model = SelectFromModel(etc, threshold = np.median(etc.feature_importances_), prefit = True)
vars_imp = transform_model.transform(vars)

# SVC rerun
x_train_imp, x_test_imp, y_train_imp, y_test_imp = train_test_split(vars_imp, resp, train_size=0.70)
svc2 = svc_gs.best_estimator_
svc2.fit(x_train_imp, y_train_imp)

print "SVC classifier with only important features\nAccuracy on training set: %.2f" % svc2.score(x_train_imp, y_train_imp)
print "The accuracy of  testing set: %.2f" % svc2.score(x_test_imp, y_test_imp)

predict_x_test_imp = svc2.predict(x_test_imp)
print "Accuracy score: %.2f" % accuracy_score(y_test_imp, predict_x_test_imp)
print "Precision score: %.2f" % precision_score(y_test_imp, predict_x_test_imp, average='weighted')
print  "Recall score: %.2f" % recall_score(y_test_imp, predict_x_test_imp, average='weighted')
print "F1 score: %.2f" % f1_score(y_test_imp, predict_x_test_imp, average='weighted')

# check out the confusion matrix & classification report
print "\nConfusion matrix: \n", confusion_matrix(y_test_imp, predict_x_test_imp)
print "\nClassification report: \n", classification_report(y_test_imp, predict_x_test_imp, target_names=resp_names)
print "-"*55

# Pruned feature set based on importance does not improve the SVC model - in fact it makes it worse!! Glass type 3 & 4 do not get identified at all!!

