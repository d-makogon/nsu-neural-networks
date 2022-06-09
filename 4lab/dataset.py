import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def train_val_test_split(data):
    train_val_len = int(len(data) * 0.8)
    test_len = len(data) - train_val_len
    train_len = int(train_val_len * 0.8)
    val_len = train_val_len - train_len

    data_train = data[:train_len]
    data_val = data[train_len : train_len + val_len]
    data_test = data[train_len + val_len : train_len + val_len + test_len]

    return data_train, data_val, data_test


def cut_df_to_dataset(data, seq_len, target_col):
    features = data.to_numpy()
    targets = data[target_col].to_numpy()

    x = sliding_window_view(features[0:-1], window_shape=(seq_len, 12))
    y = targets[seq_len:]
    x = np.squeeze(x, axis=1)
    return x.copy(), y.reshape(-1, 1).copy()
