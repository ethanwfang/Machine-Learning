import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from KDEpy import FFTKDE
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import f1_score

def feature_extraction(input, dists, fftkde_grids, measure_names, actions, sensor_names):
    # compare test samples against distributions in order to make predictions
    transformed_input = np.zeros((input.shape[0], input.shape[2] * len(measure_names) * len(actions)))

    for sample in range(input.shape[0]):
        # calculate/load the log-likelihoods
        for measure, measure_index in measure_names.items():
            for axis in range(len(sensor_names)):
                ax_sample = input[sample, :, axis]

                for action in range(len(actions)):
                    # make pdf, and use interpolation to estimate density of new points
                    y = dists[measure][action][axis]
                    custom_y = interp1d(fftkde_grids[measure_index], y, kind='linear', fill_value="extrapolate")(ax_sample)
                    transformed_input[sample, measure_index * 12 + axis*len(actions) + action] = np.sum(np.log(np.clip(custom_y, 1e-700, None)))

    return transformed_input

def smooth_predictions(predictions, window_size=5):
    if window_size >= 1 and window_size % 2 == 1:
      smoothed_predictions = []
      half_window = window_size // 2

      for i in range(len(predictions)):
          # Define the window range
          start_idx = max(0, i - half_window)
          end_idx = min(len(predictions), i + half_window + 1)

          # Get the mode of the window
          window = predictions[start_idx:end_idx]
          # Ensure scalar index works
          smoothed_label = mode(window, keepdims=True).mode[0]
          smoothed_predictions.append(smoothed_label)

      return np.array(smoothed_predictions)
    else:
      print(f'window_size={window_size} is nvalid window size')

def predict_test(train_input, train_labels, test_input):
    actions = ['rest', 'walk', 'run', 'drive']
    sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']

    train_labels -= 1

    measure_names = {
    'acc': 0
    # 'jerk': 1
    }

    train_input_jerk = np.diff(train_input, axis=1)
    train_inputs = [train_input, train_input_jerk]
    fftkde_grids = [None for _ in range(len(train_inputs))]

    for g in range(len(fftkde_grids)):
        global_min, global_max = np.min(train_inputs[g]), np.max(train_inputs[g])
        global_range = global_max - global_min
        synthetic_min = global_min - 0.15*global_range
        synthetic_max = global_max + 0.15*global_range

        # equidistant grid used during FFTKDE evaluation
        fftkde_grids[g] = np.linspace(synthetic_min, synthetic_max, num=int(global_range*100))
        # print(f"fftkde grid {g}: {fftkde_grids[g]}")

    # stores all distributions, each describing a (aix)
    distributions = {
        'acc': [[None for _ in range(len(sensor_names))] for _ in range(len(actions))]
        # 'jerk': [[None for _ in range(len(sensor_names))] for _ in range(len(actions))]
    }

    for measure, measure_index in measure_names.items():
        for axis_index, axis in enumerate(sensor_names):
            action_values = [[], [], [], []]

            # group all data for similar activites together
            for row in range(len(train_labels)):
                label = int(train_labels[row])
                action_values[label].extend(train_inputs[measure_index][row, :, axis_index])  # Add column data for this axis/table

            # Output the activity distributions for this specific axis
            for act_index, action in enumerate(actions):
                # shrink the set slightly to ensure that new values will fit strictly inside the bounds of the density function
                action_values[act_index].append(min(fftkde_grids[measure_index])*0.999)
                action_values[act_index].append(max(fftkde_grids[measure_index])*0.999)

                distributions[measure][act_index][axis_index] = FFTKDE(kernel='gaussian', bw=0.9).fit(action_values[act_index]).evaluate(fftkde_grids[measure_index])


    
    best_params_xg = {
      'colsample_bytree': 0.857,
      'gamma': 0.146,
      'learning_rate': 0.0068,
      'max_depth': 5,
      'min_child_weight': 1,
      'n_estimators': 350,
      'reg_alpha': 0.063,
      'reg_lambda': 0.183,
      'subsample': 0.753
    }

    print(f"\nStats for XGBoost")
    best_params_xg['max_depth'] = int(best_params_xg['max_depth'])
    best_params_xg['min_child_weight'] = int(best_params_xg['min_child_weight'])
    best_params_xg['n_estimators'] = int(best_params_xg['n_estimators'])

    clf = XGBClassifier(random_state=0, **best_params_xg)

    transformed_train_input = feature_extraction(train_input, distributions, fftkde_grids, measure_names, actions, sensor_names)
    clf = clf.fit(transformed_train_input, train_labels)
    predictions = clf.predict(feature_extraction(test_input, distributions, fftkde_grids, measure_names, actions, sensor_names))
    smoothed_pred = smooth_predictions(predictions, window_size=5)


    return smoothed_pred

