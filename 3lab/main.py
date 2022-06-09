import numpy as np
import matplotlib.pyplot as plt
from dataset import load_mnist, prepare_for_neural_network, random_split_train_val
from lenet import LeNet
from train import Trainer, Dataset
from optimizers import Adam
from random import seed, sample

SEED = 9231

def bc_metrics(true, predict):
    TP = ((predict == 1) & (true == 1)).sum()
    FP = ((predict == 1) & (true == 0)).sum()
    FN = ((predict == 0) & (true == 1)).sum()
    TN = ((predict == 0) & (true == 0)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (predict == true).sum() / len(predict)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, accuracy

x_train, y_train, x_test, y_test = load_mnist("data")
x_train, x_test = prepare_for_neural_network(x_train, x_test)
x_train, y_train, x_val, y_val = random_split_train_val(
    x_train, y_train, num_val=18000, random_seed=SEED)

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
    nrows=2, ncols=4, figsize=(8, 4), sharex=True, sharey=True
)
plt.rc('image', cmap='gray')
fig.suptitle('Data samples')
seed(SEED)
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
plt.clf()

print('Train size:', x_train.shape[0])
print('Val size:', x_val.shape[0])
print('Test size:', x_test.shape[0])

model = LeNet(n_output_classes=10)
dataset = Dataset(x_train, y_train, x_val, y_val)
trainer = Trainer(model, dataset, Adam(), learning_rate=1e-3,
                  num_epochs=6, batch_size=32)
loss_history, train_history = trainer.fit()
plt.plot(loss_history)
plt.savefig('loss.png')

predict = model.predict(x_test)
print(predict)

metrics = bc_metrics(predict, y_test)
print('Precision:', metrics[0])
print('Recall:', metrics[1])
print('F1:', metrics[2])
print('Accuracy:', metrics[3])
