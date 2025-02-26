# -*- coding: utf-8 -*-
"""
Script used for final evaluation of classifier accuracy on test set

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from Classify_activity import predict_test
from sklearn.model_selection import train_test_split
## 
from scipy.stats import mode
from scipy.interpolate import interp1d
from xgboost import XGBClassifier
from KDEpy import FFTKDE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_1_suffix = '_train_1.csv'
train_2_suffix = '_train_2.csv'
test_suffix = '_test.csv'

def load_sensor_data(sensor_names, suffix):
    data_slice_0 = np.loadtxt(sensor_names[0] + suffix, delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                     len(sensor_names)))
    data[:, :, 0] = data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
            sensor_names[sensor_index] + suffix, delimiter=',')   
    
    return data


    
# Load labels and sensor data into 3-D array
train_1_labels = np.loadtxt('labels' + train_1_suffix, dtype='int')
train_1_data = load_sensor_data(sensor_names, train_1_suffix)
train_2_labels = np.loadtxt('labels' + train_2_suffix, dtype='int')
train_2_data = load_sensor_data(sensor_names, train_2_suffix)
train_labels = np.hstack((train_1_labels, train_2_labels))
train_data = np.vstack((train_1_data, train_2_data))
# test_labels = np.loadtxt('labels' + test_suffix, dtype='int')
# test_data = load_sensor_data(sensor_names, test_suffix)

train_input, test_input, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=42, shuffle = False)
# Predict activities on test data
 
smoothed_pred = predict_test(train_input, train_labels, test_input)


micro_f1 = f1_score(test_labels, smoothed_pred, average='micro')
macro_f1 = f1_score(test_labels, smoothed_pred, average='macro')
print(f'Micro-averaged F1 score: {micro_f1}')
print(f'Macro-averaged F1 score: {macro_f1}')

cm = confusion_matrix(test_labels, smoothed_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1,2,3])
disp.plot()
plt.show()

# # Compute micro and macro-averaged F1 scores
# micro_f1 = f1_score(test_labels, test_outputs, average='micro')
# macro_f1 = f1_score(test_labels, test_outputs, average='macro')
# print(f'Micro-averaged F1 score: {micro_f1}')
# print(f'Macro-averaged F1 score: {macro_f1}')

# # Examine outputs compared to labels
# n_test = test_labels.size
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(n_test), test_labels, 'b.')
# plt.xlabel('Time window')
# plt.ylabel('Target')
# plt.subplot(2, 1, 2)
# plt.plot(np.arange(n_test), test_outputs, 'r.')
# plt.xlabel('Time window')
# plt.ylabel('Output (predicted target)')
# plt.show()