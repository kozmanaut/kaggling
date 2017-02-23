#!/usr/bin/env pythonimport pandas as pd

"""
Fraudulent credit card transactions - classification exercise
dataset available: https://www.kaggle.com/dalpozz/creditcardfraud
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# for prettier plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 10)

#import dataset
data = pd.read_csv('creditcard.csv')

