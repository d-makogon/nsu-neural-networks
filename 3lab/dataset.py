import numpy as np
import idx2numpy
import os

def load_mnist(folder):
    train_x = np.expand_dims(idx2numpy.convert_from_file(os.path.join(folder, "train_images")), axis=3)
    train_y = idx2numpy.convert_from_file(os.path.join(folder, "train_labels"))
    test_x = np.expand_dims(idx2numpy.convert_from_file(os.path.join(folder, "test_images")), axis=3)
    test_y = idx2numpy.convert_from_file(os.path.join(folder, "test_labels"))
    return train_x, train_y, test_x, test_y

def random_split_train_val(X, y, num_val, random_seed):
    np.random.seed(random_seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[:-num_val]
    train_X = X[train_indices]
    train_y = y[train_indices]
    val_indices = indices[-num_val:]
    val_X = X[val_indices]
    val_y = y[val_indices]
    return train_X, train_y, val_X, val_y

def prepare_for_neural_network(train_X, test_X):
    train_X = train_X.astype(np.float) / 255.0
    test_X = test_X.astype(np.float) / 255.0
    mean_image = np.mean(train_X, axis=0)
    train_X -= mean_image
    test_X -= mean_image
    return train_X, test_X
