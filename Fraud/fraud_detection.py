#!/usr/bin/env python

"""
Fraudulent credit card transactions - classification exercise
dataset available: https://www.kaggle.com/dalpozz/creditcardfraud
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# for prettier plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 10)

#import dataset
data = pd.read_csv('creditcard.csv')

# count the occurrence of each class (0: ok, 1: fraud)
pd.value_counts(data['Class'])		# 0 = 284315, 1 = 492 .... highly skewed!!

# drop the time column
data = data.drop(['Time'], axis=1)



## CLASSIFICATION
# Create x and y subsets
x = data.ix[:, data.columns != 'Class']
y = data['Class']
y_names = ['0', '1']

# Split dataset into train and test - stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify = y)


# Logistic Regression
logrc = LogisticRegression()
param = {
	"C" : [0.01, 0.1, 1,10, 100]
	}
logrc_gs = GridSearchCV(estimator = logrc, param_grid=param, cv=5, refit=True, scoring='recall')
logrc_gs.fit(x_train, y_train)
logrc = logrc_gs.best_estimator_
logrc.fit(x_train, y_train)

# Random forest
"""
Grid search:
rfc = RandomForestClassifier(n_jobs = -1, min_samples_leaf = 2)
param = {
	"n_estimators" : [10, 50, 100],
	"max_depth" : [1, 5, 10, 20]
	}
rfc_gs = GridSearchCV(estimator = rfc, param_grid=param, cv=5, refit=True, scoring='recall')
rfc_gs.fit(x_train, y_train)
rfc = rfc_gs.best_estimator_
"""
rfc = RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=10, min_samples_leaf=2)
rfc.fit(x_train, y_train)

for model in (logrc, rfc):
	print "%s:\nAccuracy on training set: %.2f" % (model, model.score(x_train, y_train))
	print "The accuracy of  testing set: %.2f" % model.score(x_test, y_test)

	predict = model.predict(x_test)
	y_score= model.predict_proba(x_test)
	fpr, tpr, thr = roc_curve(y_test, y_score[:,1])
	roc = auc(fpr, tpr)

	print "Accuracy score: %.2f" % accuracy_score(y_test, predict)
	print "Precision score: %.2f" % precision_score(y_test, predict, average='weighted')
	print  "Recall score: %.2f" % recall_score(y_test, predict, average='weighted')
	print "F1 score: %.2f" % f1_score(y_test, predict, average='weighted')

	# check out the confusion matrix & classification report
	print "\nConfusion matrix: \n", confusion_matrix(y_test, predict)
	print "\nClassification report: \n", classification_report(y_test, predict, target_names=y_names)
	cross_val_scores = cross_val_score(model, x, y, cv = 7, scoring = 'recall')
	print "Cross validation scores (recall): \n", cross_val_scores
	print "Cross validation (recall) mean: %.2f" % cross_val_scores.mean()
	print "AUC: %.3f" % roc
	print "-"*65

""""
Logistic regression classifier is pretty bad. Recall of 0.59 - miss-classifies 51/148 fraud transactions.
Random Forest is a bit better - mean CV recall rate of 0.74. Miss-classifies 42/148 fraud transactions.
RESAMPLE DATASET to have equal number of 0's and 1's!!!
"""

## Re-sample the dataset  - get the same amount of fraud and non-fraud rows
# 1. Fraud rows
fraud = data.ix[data['Class'] == 1]
# 2. Non-fraud rows
non_fraud = data.ix[data['Class'] == 0]
# 3. Randomly resample non-fraud df to the same length as the fraud df
non_fraud_subsample = non_fraud.sample(n = len(fraud))
# 4. Put the two together  in a new df
data_subsampled = pd.concat([non_fraud_subsample, fraud], ignore_index=True)

###
# Classification redo (ss = sub-sampled)
x_ss = data_subsampled.ix[:, data_subsampled.columns != 'Class']
y_ss = data_subsampled['Class']

# Split into test & train
x_ss_train, x_ss_test, y_ss_train, y_ss_test = train_test_split(x_ss, y_ss, train_size = 0.7)

# Logistic Regression
logrc_ss = LogisticRegression()
param = {
	"C" : [0.01, 0.1, 1,10, 100]
	}
logrc_ss_gs = GridSearchCV(estimator = logrc_ss, param_grid=param, cv=5, refit=True, scoring='recall')
logrc_ss_gs.fit(x_ss_train, y_ss_train)
logrc_ss = logrc_ss_gs.best_estimator_
logrc_ss.fit(x_ss_train, y_ss_train)

# Random forest
"""
#Grid search:
rfc_ss = RandomForestClassifier(n_jobs = -1, min_samples_leaf = 2)
param = {
	"n_estimators" : [10, 50, 100],
	"max_depth" : [1, 5, 10, 20]
	}
rfc_ss_gs = GridSearchCV(estimator = rfc_ss, param_grid=param, cv=5, refit=True, scoring='recall')
rfc_ss_gs.fit(x_ss_train, y_ss_train)
rfc_ss = rfc_ss_gs.best_estimator_
"""
rfc_ss = RandomForestClassifier(criterion='gini', max_depth=10, n_estimators=50, min_samples_leaf=2)
rfc_ss.fit(x_ss_train, y_ss_train)

for model in (logrc_ss, rfc_ss):
	print "%s:\nAccuracy on training set: %.2f" % (model, model.score(x_ss_train, y_ss_train))
	print "The accuracy of  testing set: %.2f" % model.score(x_ss_test, y_ss_test)

	predict = model.predict(x_ss_test)
	y_score= model.predict_proba(x_ss_test)
	fpr, tpr, thr = roc_curve(y_ss_test, y_score[:,1])
	roc = auc(fpr, tpr)

	print "Accuracy score: %.2f" % accuracy_score(y_ss_test, predict)
	print "Precision score: %.2f" % precision_score(y_ss_test, predict, average='weighted')
	print  "Recall score: %.2f" % recall_score(y_ss_test, predict, average='weighted')
	print "F1 score: %.2f" % f1_score(y_ss_test, predict, average='weighted')

	# check out the confusion matrix & classification report
	print "\nConfusion matrix: \n", confusion_matrix(y_ss_test, predict)
	print "\nClassification report: \n", classification_report(y_ss_test, predict, target_names=y_names)
	cross_val_scores = cross_val_score(model, x_ss, y_ss, cv = 7, scoring = 'recall')
	print "Cross validation scores (recall): \n", cross_val_scores
	print "Cross validation (recall) mean: %.2f" % cross_val_scores.mean()
	print "AUC: %.3f" % roc
	print "-"*65

# Much better results! RF ever so slightly better than LogRC - fraud detected 133/147 vs 132/147. But non fraud detected by one worse: 141/149 [RF] vs 142/149 [LogRC]

####
# See if normalizing the 'amount' column will change anything
data_subsampled['AmountNorm'] = StandardScaler().fit_transform(data_subsampled['Amount'].reshape(-1,1))
data_subsampled = data_subsampled.drop(['Amount'], axis = 1)

x_norm = data_subsampled.ix[:, data_subsampled.columns != 'Class']
y_norm = data_subsampled['Class']

x_norm_train, x_norm_test, y_norm_train, y_norm_test = train_test_split(x_norm, y_norm, train_size = 0.7)

# Logistic Regression
logrc_norm = LogisticRegression()
param = {
	"C" : [0.01, 0.1, 1,10, 100]
	}
logrc_norm_gs = GridSearchCV(estimator = logrc_norm, param_grid=param, cv=5, refit=True, scoring='recall')
logrc_norm_gs.fit(x_norm_train, y_norm_train)
logrc_norm = logrc_norm_gs.best_estimator_
logrc_norm.fit(x_norm_train, y_norm_train)

# Random forest
"""
#Grid search:
rfc_norm = RandomForestClassifier(n_jobs = -1, min_samples_leaf = 2)
param = {
	"n_estimators" : [5, 10, 50, 100],
	"max_depth" : [1, 5, 10, 20]
	}
rfc_norm_gs = GridSearchCV(estimator = rfc_norm, param_grid=param, cv=5, refit=True, scoring='recall')
rfc_norm_gs.fit(x_norm_train, y_norm_train)
rfc_norm = rfc_norm_gs.best_estimator_
"""
rfc_norm = RandomForestClassifier(criterion='gini', max_depth=10, n_estimators=5, min_samples_leaf=2)
rfc_norm.fit(x_norm_train, y_norm_train)

for model in (logrc_norm, rfc_norm):
	print "%s:\nAccuracy on training set: %.2f" % (model, model.score(x_norm_train, y_norm_train))
	print "The accuracy of  testing set: %.2f" % model.score(x_norm_test, y_norm_test)

	predict = model.predict(x_norm_test)
	y_score= model.predict_proba(x_norm_test)
	fpr, tpr, thr = roc_curve(y_norm_test, y_score[:,1])
	roc = auc(fpr, tpr)

	print "Accuracy score: %.2f" % accuracy_score(y_norm_test, predict)
	print "Precision score: %.2f" % precision_score(y_norm_test, predict, average='weighted')
	print  "Recall score: %.2f" % recall_score(y_norm_test, predict, average='weighted')
	print "F1 score: %.2f" % f1_score(y_norm_test, predict, average='weighted')

	# check out the confusion matrix & classification report
	print "\nConfusion matrix: \n", confusion_matrix(y_norm_test, predict)
	print "\nClassification report: \n", classification_report(y_norm_test, predict, target_names=y_names)
	cross_val_scores = cross_val_score(model, x_norm, y_ss, cv = 7, scoring = 'recall')
	print "Cross validation scores (recall): \n", cross_val_scores
	print "Cross validation (recall) mean: %.2f" % cross_val_scores.mean()
	print "AUC: %.3f" % roc
	print "-"*65

## Results: Normalization improves things! Recall rate slightly up. AUC for LogRC now 0.981 (vs 0.968!). AUC for RF now 0.976 (vs 0.972)
