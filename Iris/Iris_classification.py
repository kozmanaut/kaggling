""" Machine learning classification using the iris dataset """

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklear.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



# load dataset and separate variables
iris = load_iris()
x = iris.data()
y = iris.target()

# split the variables into train and testing subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=5)

# scale the data
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

# Find best parameters for SVC using the GridSearchCV
clf = SVC(probability=True)
param = {
	"C" : [0.1,1,10,100],
	"kernel" : ['rbf', 'poly'],
	"degree" : [2, 4, 6],
	"gamma" : ['auto', 1, 5, 10]
	}
gs = GridSearchCV(estimator = clf, param_grid=param, cv=5, refit=True, scoring='accuracy')
gs.fit(x_train_std, y_train)
clf = gs.best_estimator_
clf.fit(x_train_std, y_train)

# find how well the model performs
clf.score(x_train_std, y_train)	# 0.971
clf.score(x_test_std, y_test)	# 0.955

predict_x_test = clf.predict(x_test_std)
accuracy_score(y_test, predict_x_test)				# 0.955
precision_score(y_test, predict_x_test, average='weighted') 	# 0.955
recall_score(y_test, predict_x_test, average='weighted')	# 0.955
f1_score(y_test, predict_x_test, average='weighted')		# 0.955

# check out the confusion matrix & classification report
confusion_matrix(y_test, predict_x_test)

print(classification_report(y_test, predict_x_test, target_names=iris.target_names))
