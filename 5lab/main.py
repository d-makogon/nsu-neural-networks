import os
from random import sample, seed
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from model import GAN

def load_mnist(folder):
    train_x = np.expand_dims(
        idx2numpy.convert_from_file(os.path.join(folder, "train_images")), axis=3
    )
    train_y = idx2numpy.convert_from_file(os.path.join(folder, "train_labels"))
    test_x = np.expand_dims(
        idx2numpy.convert_from_file(os.path.join(folder, "test_images")), axis=3
    )
    test_y = idx2numpy.convert_from_file(os.path.join(folder, "test_labels"))
    return train_x, train_y, test_x, test_y


def random_split_train_val(X, y, random_seed):
    np.random.seed(random_seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:]
    train_X = X[train_indices]
    train_y = y[train_indices]

    return train_X, train_y

x_train,y_train, x_test, y_test = load_mnist("data")
x_train, y_train = random_split_train_val(x_train, y_train, random_seed=93123)

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
    nrows=2, ncols=4, figsize=(8, 4), sharex=True, sharey=True
)
plt.rc('image', cmap='gray')
fig.suptitle('Train and test data samples')
seed(93123)
train_samples = sample(range(len(x_train)), 4)
test_samples = sample(range(len(x_test)), 4)
ax1.imshow(x_train[train_samples[0]][:,:,::-1])
ax1.axis("off")
ax2.imshow(x_train[train_samples[1]][:,:,::-1])
ax2.axis("off")
ax3.imshow(x_train[train_samples[2]][:,:,::-1])
ax3.axis("off")
ax4.imshow(x_train[train_samples[3]][:,:,::-1])
ax4.axis("off")
ax5.imshow(x_test[test_samples[0]][:,:,::-1])
ax5.axis("off")
ax6.imshow(x_test[test_samples[1]][:,:,::-1])
ax6.axis("off")
ax7.imshow(x_test[test_samples[2]][:,:,::-1])
ax7.axis("off")
ax8.imshow(x_test[test_samples[3]][:,:,::-1])
ax8.axis("off")
plt.show()
plt.savefig('data.png')
print('Train size:', x_train.shape[0])
print('Test size:', x_test.shape[0])
model = GAN([7], epochs=100, learning_rate=1e-3, decay_rate=1e-4)
J_Ds, J_Gs = model.train(x_train, y_train)