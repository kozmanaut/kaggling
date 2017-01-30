""" Machine learning classification using the iris dataset """

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# load dataset and separate variables
iris = load_iris()
x = iris.data
y = iris.target

# split the variables into train and testing subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=5)

# scale the data
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

# Find best parameters for SVC using the GridSearchCV
svc = SVC(probability=True, random_state=5)
param = {
	"C" : [0.1,1,10,100],
	"kernel" : ['rbf', 'poly'],
	"degree" : [2, 4, 6],
	"gamma" : ['auto', 1, 5, 10]
	}
svc_gs = GridSearchCV(estimator = svc, param_grid=param, cv=5, refit=True, scoring='accuracy')
svc_gs.fit(x_train_std, y_train)
svc = svc_gs.best_estimator_
svc.fit(x_train_std, y_train)

# find how well the model performs
print "SVC classifier\nAccuracy on training set: %.2f" % svc.score(x_train_std, y_train)

print "The accuracy of  testing set: %.2f" % svc.score(x_test_std, y_test)

predict_x_test = svc.predict(x_test_std)
print "Accuracy score: %.2f" % accuracy_score(y_test, predict_x_test)

print "Precision score: %.2f" % precision_score(y_test, predict_x_test, average='weighted')

print  "Recall score: %.2f" % recall_score(y_test, predict_x_test, average='weighted')

print "F1 score: %.2f" % f1_score(y_test, predict_x_test, average='weighted')

# check out the confusion matrix & classification report
print "\nConfusion matrix: \n", confusion_matrix(y_test, predict_x_test)

print "\nClassification report: \n", classification_report(y_test, predict_x_test, target_names=iris.target_names)
print "-"*55

# Conclusion: SVC performs well. Only one mis-classified instance


################
# Try KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier( )
knc.fit(x_train_std, y_train)

# Check accuracy, confusion matrix & classification report of KN classifier
print "KNeighborsClassifier\nAccuracy score on training set: %.2f" % knc.score(x_train_std, y_train)
print "Accuracy score on testing set: %.2f" % knc.score(x_test_std, y_test)

predict_knc = knc.predict(x_test_std)

print "\nConfusion matrix: \n", confusion_matrix(y_test, predict_knc)
print "\nClassification report: \n", classification_report(y_test, predict_knc, target_names=iris.target_names)
print "-"*55

# Conclusions: KNC performs a little worse than SVC - mis-classifies 2 versicolor as virginica


##############
## Try Random forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
param = {
	"n_estimators" : [5, 10, 50, 100],
	"criterion" : ['gini', 'entropy'],
	"max_depth" : [5, 10, 15, 20],
	"class_weight": [None, "balanced"]
	}
rfc_gs = GridSearchCV(estimator = rfc, param_grid=param, cv=5, refit=True, scoring='accuracy')
rfc_gs.fit(x_train_std, y_train)
rfc = rfc_gs.best_estimator_
rfc.fit(x_train_std, y_train)

# Check accuracy, confusion matrix & classification report of RF classifier
print "RandomForestClassifier\nAccuracy score on training set: %.2f" % rfc.score(x_train_std, y_train)
print "Accuracy score on testing set: %.2f" % rfc.score(x_test_std, y_test)

predict_rfc = rfc.predict(x_test_std)

print "\nConfusion matrix: \n", confusion_matrix(y_test, predict_rfc)
print "\nClassification report: \n", classification_report(y_test, predict_rfc, target_names=iris.target_names)
print "-"*55

# Conclusion: RFC performs a little worse than SVC or KNC, mis-classifies 3 instances. Overfitting? Accuracy on training set = 100%


###########
# Try GaussianNB - Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB

nbc = GaussianNB()
nbc.fit(x_train_std, y_train)

# Check accuracy, confusion matrix & classification report of NB classifier
print "NaiveBayesClassifier\nAccuracy score on training set: %.2f" % nbc.score(x_train_std, y_train)
print "Accuracy score on testing set: %.2f" % nbc.score(x_test_std, y_test)

predict_nbc = nbc.predict(x_test_std)

print "\nConfusion matrix: \n", confusion_matrix(y_test, predict_nbc)
print "\nClassification report: \n", classification_report(y_test, predict_nbc, target_names=iris.target_names)
print "-"*55

# Conclusion: NBC performs as poorly as RFC - 3 mis-classified instances. No such severe overfitting though.

########
# Try PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_pca3 = PCA(n_components = 3).fit_transform(x)

fig = plt.figure(1, figsize = (10,10))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(x_pca3[:, 0], x_pca3[:, 1], x_pca3[:, 2], c= y)
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()

x_pca2 = PCA(n_components = 2).fit_transform(x)
plt.scatter(x=x_pca2[:, 0], y= x_pca2[:, 1], c = y, alpha = 0.6)
plt.xlabel('1st eigenvector')
plt.xticks([])
plt.ylabel('2nd eigenvector')
plt.yticks([])
plt.title('PCA of Iris dataset using two components')
plt.tight_layout()
plt.show()