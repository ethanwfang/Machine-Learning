# -*- coding: utf-8 -*-
"""
Script used to evaluate classifier accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from classifySpam import predictTest
# 
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression



desiredFPR = 0.01
train1DataFilename = '/Users/efang/Desktop/coding/Intro-to-ML/CSDS340/data/spamTrain1.csv'
train2DataFilename = '/Users/efang/Desktop/coding/Intro-to-ML/CSDS340/data/spamTrain2.csv'

def tprAtFPR(labels, outputs, desiredFPR):
    fpr, tpr, thres = roc_curve(labels, outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = ((tprAbove - tprBelow) / (fprAbove - fprBelow) * (desiredFPR - fprBelow)
             + tprBelow)
    return tprAt, fpr, tpr

# Load the training datasets
print("loading training data... ")
train1Data = np.loadtxt(train1DataFilename, delimiter=',')
train2Data = np.loadtxt(train2DataFilename, delimiter=',')

# Combine the two datasets
print("combining datasets...")
trainData = np.r_[train1Data, train2Data]

# Split features and labels
trainFeatures = trainData[:, :-1]  # All columns except the last one are features
trainLabels = trainData[:, -1]     # The last column is the label

# Perform a train-test split on the combined dataset
train_input, test_input, train_output, test_output = train_test_split(
    trainFeatures,
    trainLabels,
    test_size=0.1,     
    random_state=1    
)

# Predict test set outputs
print("Starting predictions...")
testOutputs = predictTest(train_input, train_output, test_input)

# Evaluate model performance
print("evaluating performance...")
aucTestRun = roc_auc_score(test_output, testOutputs)
tprAtDesiredFPR, fpr, tpr = tprAtFPR(test_output, testOutputs, desiredFPR)

print("plotting curve...")
# Plot ROC curve
plt.plot(fpr, tpr)
print(f'Test set AUC: {aucTestRun}')
print(f'TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for spam detector')    
plt.show()
