import numpy as np
from copy import deepcopy
from layers import softmax_with_cross_entropy


def multiclass_accuracy(true, predict):
    return (predict == true).sum() / len(predict)

class Dataset:
    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y

class Trainer:
    def __init__(
        self,
        model,
        dataset,
        optim,
        num_epochs=20,
        batch_size=20,
        learning_rate=1e-3,
        learning_rate_decay=1.0,
    ):
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.learning_rate_decay = learning_rate_decay
        self.optimizers = None

    def get_data(self, b_ind):
        b_X = self.dataset.train_X[b_ind]
        b_X_g = np.asarray(b_X)
        b_Y = self.dataset.train_y[b_ind]
        return b_X, b_Y, b_X_g

    def fit(self):
        if self.optimizers is None:
            self.setup_optimizers()
        num_train = self.dataset.train_X.shape[0]
        loss_history = []
        train_acc_history = []
        for epoch in range(self.num_epochs):
            shuffled = np.arange(num_train)
            np.random.shuffle(shuffled)
            sc = np.arange(self.batch_size, num_train, self.batch_size)
            bs_ind = np.array_split(shuffled, sc)
            b_losses = []
            correct = 0
            for _idx, b_ind in enumerate(bs_ind):
                b_X, b_Y, b_X_g = self.get_data(b_ind)
                for p in self.model.params().values():
                    p.grad = 0
                o = self.model.forward(b_X_g)
                l, g = softmax_with_cross_entropy(o, b_Y)
                pred = np.argmax(o, axis=1)
                correct += np.sum(pred == b_Y)
                self.model.backward(g)
                for param_name, p in self.model.params().items():
                    opt = self.optimizers[param_name]
                    p.value = opt.update(
                        p.value, p.grad, self.learning_rate
                    )
                b_losses.append(l)
            self.learning_rate *= self.learning_rate_decay
            ave_loss = np.mean(b_losses)
            train_accuracy = correct / len(self.dataset.train_X)
            print(
                f"Epoch #{epoch+1}: Loss = {b_losses[-1]:.3f}, accuracy = {train_accuracy:.3f}"
            )
            loss_history.append(ave_loss)
            train_acc_history.append(train_accuracy)
        return loss_history, train_acc_history

    def setup_optimizers(self):
        params = self.model.params()
        self.optimizers = {}
        for param_name, param in params.items():
            self.optimizers[param_name] = deepcopy(self.optim)

    def compute_accuracy(self, X, y):
        inds = np.arange(X.shape[0])
        sc = np.arange(self.batch_size, X.shape[0], self.batch_size)
        b_inds = np.array_split(inds, sc)
        p = np.zeros_like(y)
        for b_ind in b_inds:
            b_X = X[b_ind]
            b_X_g = np.asarray(b_X)
            pred_b = self.model.predict(b_X_g)
            p[b_ind] = pred_b
        return multiclass_accuracy(p, y)
