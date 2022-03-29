import pandas as pd
import numpy as np


def global_preprocessing(X_data, leak_labels, time_start, time_win_len, sensors_idx=None):
    X = []
    Y = []
    y_leak = []
    
    if sensors_idx is None:
        sensors_idx = list(range(X_data.shape[1]))
    
     # Use a sliding time window to construct a labeled data set
    t_index = time_start
    time_points = range(len(leak_labels))
    i = 0
    while t_index < len(time_points) - time_win_len:
        # Grab time window from data stream
        x = X_data[t_index:t_index+time_win_len-1, sensors_idx]

        #######################
        # Feature engineering #
        #######################
        x = np.mean(x,axis=0)  # "Stupid" feature
        X.append(x)

        Y.append([X_data[t_index + time_win_len-1, n] for n in sensors_idx])

        y_leak.append(leak_labels[t_index + time_win_len-1])

        t_index += 1  # Note: Overlapping time windows
        i += 1

    X = np.array(X)
    Y = np.array(Y)
    y_leak = np.array(y_leak)

    return X, Y, y_leak
