#!/usr/bin/env python

""" Glass identification exercise: https://www.kaggle.com/uciml/glass"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
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
x_train, x_test, y_train, y_test = train_test_split(vars_std, resp, train_size=0.7, stratify = resp)

####
#SVC

svc = SVC(probability=True)
param = {
	"C" : [1,10,100],
	"kernel" : ['rbf', 'poly'],
	"degree" : [2, 4, 6],
	"gamma" : ['auto', 1, 5, 10]
	}
svc_gs = GridSearchCV(estimator = svc, param_grid=param, cv=5, refit=True, scoring='accuracy')
svc_gs.fit(x_train, y_train)
svc = svc_gs.best_estimator_

#svc = SVC(C = 1, kernel = 'rbf', probability = True, degree = 2, gamma = 'auto')
svc.fit(x_train, y_train)


############
# Random Forest

#GRID SEARCH:
rfc = RandomForestClassifier(warm_start=True, n_jobs = -1)
param = {
	"n_estimators" : [10, 50, 100],
	"min_samples_leaf" : [1, 2, 10],
	"criterion" : ['gini', 'entropy'],
	"max_depth" : [10, 15, 20],
	"class_weight": [None, "balanced"]
	}
rfc_gs = GridSearchCV(estimator = rfc, param_grid=param, cv=5, refit=True, scoring='accuracy')
rfc_gs.fit(x_train, y_train)
rfc = rfc_gs.best_estimator_

#rfc = RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=100, min_samples_leaf=2)
rfc.fit(x_train, y_train)


#############
# AdaBoost

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=3), n_estimators=100)
abc.fit(x_train, y_train)


######
# KNeighborsClassifier (with all the scaled features)

knc = KNeighborsClassifier()
param = {
	"n_neighbors" : [1, 2, 5, 10],
	"weights" : ['uniform', 'distance'],
}
knc_gs = GridSearchCV(estimator = knc, param_grid=param, cv=5, refit=True, scoring='accuracy')
knc_gs.fit(x_train, y_train)
knc = knc_gs.best_estimator_
knc.fit(x_train, y_train)

for model in (svc, rfc, abc, knc):
	print "%s:\nAccuracy on training set: %.2f" % (model, model.score(x_train, y_train))
	print "The accuracy of  testing set: %.2f" % model.score(x_test, y_test)

	predict_x_test = model.predict(x_test)
	print "Accuracy score: %.2f" % accuracy_score(y_test, predict_x_test)
	print "Precision score: %.2f" % precision_score(y_test, predict_x_test, average='weighted')
	print  "Recall score: %.2f" % recall_score(y_test, predict_x_test, average='weighted')
	print "F1 score: %.2f" % f1_score(y_test, predict_x_test, average='weighted')

	# check out the confusion matrix & classification report
	print "\nConfusion matrix: \n", confusion_matrix(y_test, predict_x_test)
	print "\nClassification report: \n", classification_report(y_test, predict_x_test, target_names=resp_names)
	print "Cross validation scores: \n", cross_val_score(model, vars_std, resp, cv = 7, scoring = 'accuracy')
	print "Cross validation mean: %.2f" % cross_val_score(model, vars_std, resp, cv = 7, scoring = 'accuracy').mean()
	print "-"*65

## Random Forest perfroms best with ~73% accuracy - not too great

##############################
## TRY PCA dimension reduction and run SVC, RF & KNC
from sklearn.decomposition import PCA

vars_pca = PCA(n_components = 3).fit_transform(vars_std)

# split to train & test datasets
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(vars_pca, resp, stratify = resp, train_size=0.70)
# SVC
svc_pca = SVC(C = 100, kernel = 'rbf', probability = True, degree = 2, gamma = 'auto')
svc_pca.fit(x_train_pca, y_train_pca)

#RF
rfc_pca = RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=100, warm_start=True, min_samples_leaf=2)
rfc_pca.fit(x_train_pca, y_train_pca)

#ABC
abc_pca = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=3), n_estimators=100)
abc_pca.fit(x_train_pca, y_train_pca)

#KNC
knc_pca = KNeighborsClassifier(n_neighbors = 2, weights = 'uniform')
knc_pca.fit(x_train_pca, y_train_pca)

for model in (svc_pca, rfc_pca, abc_pca, knc_pca):
	print "%s:\nAccuracy on training set: %.2f" % (model, model.score(x_train_pca, y_train_pca))
	print "The accuracy of  testing set: %.2f" % model.score(x_test_pca, y_test_pca)

	predict_x_test_pca = model.predict(x_test_pca)
	print "Accuracy score: %.2f" % accuracy_score(y_test_pca, predict_x_test_pca)
	print "Precision score: %.2f" % precision_score(y_test_pca, predict_x_test_pca, average='weighted')
	print  "Recall score: %.2f" % recall_score(y_test_pca, predict_x_test_pca, average='weighted')
	print "F1 score: %.2f" % f1_score(y_test_pca, predict_x_test_pca, average='weighted')

	# check out the confusion matrix & classification report
	print "\nConfusion matrix: \n", confusion_matrix(y_test_pca, predict_x_test_pca)
	print "\nClassification report: \n", classification_report(y_test_pca, predict_x_test_pca, target_names=resp_names)
	print "Cross validation scores: \n", cross_val_score(model, vars_pca, resp, cv = 7, scoring = 'accuracy')
	print "Cross validation mean: %.2f" % cross_val_score(model, vars_pca, resp, cv = 7, scoring = 'accuracy').mean()
	print "-"*65

## Dimensionality reduction with PCA does not improve analysis

#############
# Reduce number of features  - dimensionality reduction [based on feature importance] and rerun
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# Feature importance
etc = ExtraTreesClassifier(n_estimators = 250)
etc = etc.fit(vars, resp)
print "Dimensionality Reduction\n\Feature importance: \n", etc.feature_importances_
transform_model = SelectFromModel(etc, threshold = np.median(etc.feature_importances_), prefit = True)
vars_imp = transform_model.transform(vars)

# SVC rerun
x_train_imp, x_test_imp, y_train_imp, y_test_imp = train_test_split(vars_imp, resp, train_size=0.70)
svc2 = svc_gs.best_estimator_
svc2.fit(x_train_imp, y_train_imp)

#RFC
rfc2 = rfc_gs.best_estimator_
rfc2.fit(x_train_imp, y_train_imp)

#ABC
abc2 = abc
abc2.fit(x_train_imp, y_train_imp)

#KNC
knc2 = knc_gs.best_estimator_
knc2 = knc2.fit(x_train_imp, y_train_imp)

for model in (svc2, rfc2, abc2, knc2):
	print "%s:\nAccuracy on training set: %.2f" % (model, model.score(x_train_imp, y_train_imp))
	print "The accuracy of  testing set: %.2f" % model.score(x_test_imp, y_test_imp)

	predict_x_test_imp = model.predict(x_test_imp)
	print "Accuracy score: %.2f" % accuracy_score(y_test_imp, predict_x_test_imp)
	print "Precision score: %.2f" % precision_score(y_test_imp, predict_x_test_imp, average='weighted')
	print  "Recall score: %.2f" % recall_score(y_test_imp, predict_x_test_imp, average='weighted')
	print "F1 score: %.2f" % f1_score(y_test_imp, predict_x_test_imp, average='weighted')

	# check out the confusion matrix & classification report
	print "\nConfusion matrix: \n", confusion_matrix(y_test_imp, predict_x_test_imp)
	print "\nClassification report: \n", classification_report(y_test_imp, predict_x_test_imp, target_names=resp_names)
	print "Cross validation scores: \n", cross_val_score(model, vars_imp, resp, cv = 7, scoring = 'accuracy')
	print "Cross validation mean: %.2f" % cross_val_score(model, vars_imp, resp, cv = 7, scoring = 'accuracy').mean()
	print "-"*65

# Dimensionality reduction based on importance does not improve the models - about the same accuracy


###########################################################################
"""
Glass type 6 has only 9 instances - remove from analysis and reapply models
"""

glass_n6 = glass.loc[glass['Type'] != 6]

# Convert to numpy arrays
vars_n6 = glass_n6.as_matrix(columns=variable_cols)
resp_n6  = glass_n6.as_matrix(columns = ['Type']).ravel()
resp_names_n6 = map(str, np.unique(resp_n6).tolist())

# Standardise & plot
sc = StandardScaler()
vars_std_n6 = sc.fit_transform(vars_n6)

# Split data into train and test
x_train_n6, x_test_n6, y_train_n6, y_test_n6 = train_test_split(vars_std_n6, resp_n6, train_size=0.7)

####
#SVC

svc_n6 = SVC(probability=True)
param = {
	"C" : [1,10,100],
	"kernel" : ['rbf', 'poly'],
	"degree" : [2, 4, 6],
	"gamma" : ['auto', 1, 5, 10]
	}
svc_n6_gs = GridSearchCV(estimator = svc_n6, param_grid=param, cv=5, refit=True, scoring='accuracy')
svc_n6_gs.fit(x_train_n6, y_train_n6)
svc_n6 = svc_n6_gs.best_estimator_

#svc_n6 = SVC(C = 1, kernel = 'rbf', probability = True, degree = 2, gamma = 'auto')
svc_n6.fit(x_train_n6, y_train_n6)


############
# Random Forest

#GRID SEARCH:
rfc_n6 = RandomForestClassifier(warm_start=True, n_jobs = -1)
param = {
	"n_estimators" : [10, 50, 100],
	"min_samples_leaf" : [1, 2, 10],
	"criterion" : ['gini', 'entropy'],
	"max_depth" : [1, 2, 10, 20],
	"class_weight": [None, "balanced"]
	}
rfc_n6_gs = GridSearchCV(estimator = rfc_n6, param_grid=param, cv=5, refit=True, scoring='accuracy')
rfc_n6_gs.fit(x_train_n6, y_train_n6)
rfc_n6 = rfc_n6_gs.best_estimator_

#rfc_n6 = RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=100, min_samples_leaf=2)
rfc_n6.fit(x_train_n6, y_train_n6)


#############
# AdaBoost

abc_n6 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=3), n_estimators=100)
abc_n6.fit(x_train_n6, y_train_n6)


######
# KNeighborsClassifier (with all the scaled features)

knc_n6 = KNeighborsClassifier()
param = {
	"n_neighbors" : [1, 2, 5, 10],
	"weights" : ['uniform', 'distance'],
}
knc_n6_gs = GridSearchCV(estimator = knc_n6, param_grid=param, cv=5, refit=True, scoring='accuracy')
knc_n6_gs.fit(x_train_n6, y_train_n6)
knc_n6 = knc_n6_gs.best_estimator_
knc_n6.fit(x_train_n6, y_train_n6)

#######
# Gradient Boost
gbc_n6 = GradientBoostingClassifier()
param = {
	"n_estimators" : [50, 100, 150, 200],
	"learning_rate" : [0.1, 0.2, 0.5, 1],
	"max_depth" : [1, 2],
	"min_samples_leaf" : [1, 2, 3],
}
gbc_n6_gs = GridSearchCV(estimator = gbc_n6, param_grid = param, cv = 5, refit = True, scoring = 'accuracy')
gbc_n6_gs.fit(x_train_n6, y_train_n6)
gbc_n6 = gbc_n6_gs.best_estimator_
gbc_n6.fit(x_train_n6, y_train_n6)

for model in (svc_n6, rfc_n6, abc_n6, knc_n6, gbc_n6):
	print "%s:\nAccuracy on training set: %.2f" % (model, model.score(x_train_n6, y_train_n6))
	print "The accuracy of  testing set: %.2f" % model.score(x_test_n6, y_test_n6)

	predict_x_test_n6 = model.predict(x_test_n6)
	print "Accuracy score: %.2f" % accuracy_score(y_test_n6, predict_x_test_n6)
	print "Precision score: %.2f" % precision_score(y_test_n6, predict_x_test_n6, average='weighted')
	print  "Recall score: %.2f" % recall_score(y_test_n6, predict_x_test_n6, average='weighted')
	print "F1 score: %.2f" % f1_score(y_test_n6, predict_x_test_n6, average='weighted')

	# check out the confusion matrix & classification report
	print "\nConfusion matrix: \n", confusion_matrix(y_test_n6, predict_x_test_n6)
	print "\nClassification report: \n", classification_report(y_test_n6, predict_x_test_n6, target_names=resp_names_n6)
	print "Cross validation scores: \n", cross_val_score(model, vars_std_n6, resp_n6, cv = 7, scoring = 'accuracy')
	print "Cross validation mean: %.2f" % cross_val_score(model, vars_std_n6, resp_n6, cv = 7, scoring = 'accuracy').mean()
	print "-"*65
