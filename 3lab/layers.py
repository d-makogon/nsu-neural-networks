import numpy as np

def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W
    return loss, grad

def softmax(predictions):
    if len(predictions.shape) == 2:
        s = np.max(predictions, axis=1)
        e_x = np.exp(predictions - s[:, np.newaxis])
        div = np.sum(e_x, axis=1)
        return e_x / div[:, np.newaxis]
    else:
        exps = np.exp(predictions - np.max(predictions))
        return exps / np.sum(exps)

def cross_entropy_loss(probs, target_index):
    if type(target_index) == int:
        target_index = np.asarray([target_index])
    if len(target_index.shape) == 1:
        probs = probs[target_index]
    else:
        probs = probs[range(probs.shape[0]), target_index[:, 0]]
    probs = np.clip(probs, 0.001, 1)
    return -np.mean(np.log(probs))

def softmax_with_cross_entropy(preds, target_index):
    if type(target_index) == int:
        index = np.asarray([target_index])
    else:
        index = target_index.copy()
    if index.ndim == 1 and index.size > 1:
        index = index.reshape(-1, 1)
    prob = softmax(preds.copy())
    loss = cross_entropy_loss(prob, index)
    y = np.zeros_like(preds)
    if len(index.shape) == 1:
        y[index] = 1
    else:
        y[range(y.shape[0]), index[:, 0]] = 1
    dprediction = prob - 1 * y
    if preds.ndim == 2:
        dprediction = dprediction / preds.shape[0]
    return loss, dprediction

class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
def sigmoid(X):
    return 1 / (1 + np.exp(-X))
def sigmoid_backward(X, d_out):
    return d_out * (sigmoid(X) * (1 - sigmoid(X)))

class Sigmoid:
    def __init__(self):
        self.name = "sigmoid"
    def forward(self, X):
        self.X = X
        return sigmoid(X)
    def backward(self, d_out):
        return sigmoid_backward(self.X, d_out)
    def params(self):
        return {}

def fc_forward(X, W, B):
    return np.dot(X, W) + B

def fc_backward(X, d_out, W):
    W_grad = np.dot(X.T, d_out)
    B_grad = np.array([np.sum(d_out, axis=0)])
    gradX = np.dot(d_out, W.T)
    return (W_grad, B_grad, gradX)

class FullyConnected:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        self.name = "FC"
    def forward(self, X):
        self.X = X
        return fc_forward(X, self.W.value, self.B.value)
    def backward(self, d_out):
        r = fc_backward(self.X, d_out, self.W.value)
        self.W.grad = r[0]
        self.B.grad = r[1]
        return r[2]
    def params(self):
        return {"W": self.W, "B": self.B}

class Convolutional:
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1):
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size, in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        self.padding = padding
        self.stride = stride
        self.name = "conv"
        self.out = None
        self.grad_X = None
    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = 0
        out_width = 0
        self.X = X.copy()
        out_height = (
            np.floor(height + 2 * self.padding - self.filter_size) / self.stride + 1
        ).astype(np.int32)
        out_width = (
            np.floor(width + 2 * self.padding - self.filter_size) / self.stride + 1
        ).astype(np.int32)
        if self.out is None or self.out.shape[0] != batch_size:
            self.out = np.zeros([batch_size, out_height, out_width, self.out_channels])
        self.X = np.pad(
            array=self.X,
            pad_width=(
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ),
            mode="constant",
        )
        slice_W = self.W.value.reshape(-1, self.out_channels)
        for y in range(0, out_height):
            for x in range(0, out_width):
                slice_X = self.X[
                    :,
                    y * self.stride : y * self.stride + self.filter_size,
                    x * self.stride : x * self.stride + self.filter_size,
                    :,
                ].reshape(batch_size, -1)
                self.out[:, y, x, :] = slice_X.dot(slice_W) + self.B.value
                pass
        return self.out
    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        slice_W = self.W.value.reshape(-1, self.out_channels)
        if self.grad_X is None or not np.array_equal(self.grad_X.shape, self.X):
            grad_X = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                slice_X = self.X[
                    :,
                    y * self.stride : y * self.stride + self.filter_size,
                    x * self.stride : x * self.stride + self.filter_size,
                    :,
                ].reshape(batch_size, -1)
                self.W.grad += np.dot(slice_X.T, d_out[:, y, x, :]).reshape(
                    (self.filter_size, self.filter_size, self.in_channels, out_channels)
                )
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)
                grad_X[
                    :,
                    y * self.stride : y * self.stride + self.filter_size,
                    x * self.stride : x * self.stride + self.filter_size,
                    :,
                ] += np.dot(d_out[:, y, x, :], slice_W.T).reshape(
                    (batch_size, self.filter_size, self.filter_size, self.in_channels)
                )
                pass
        return grad_X[
            :,
            self.padding : height - self.padding,
            self.padding : width - self.padding,
            :,
        ]
    def params(self):
        return {"W": self.W, "B": self.B}

class MaxPooling:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.name = "maxpooling"
        self.out = None
        self.result = None
        self.tmp_result = None
    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        if self.out is None or not np.array_equal(
            self.out.shape, [batch_size, out_height, out_width, channels]
        ):
            self.out = np.zeros([batch_size, out_height, out_width, channels])
        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride
                self.out[:, y, x, :] = np.max(
                    X[
                        :,
                        y_stride : y_stride + self.pool_size,
                        x_stride : x_stride + self.pool_size,
                        :,
                    ],
                    axis=(1, 2),
                )
        return self.out
    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        if self.result is None or not np.array_equal(self.result.shape, self.X.shape):
            self.result = np.zeros(self.X.shape)
        batch_inds, channels_inds = np.repeat(range(batch_size), channels), np.tile(
            range(channels), batch_size
        )
        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride
                slice_X = self.X[
                    :,
                    y_stride : y_stride + self.pool_size,
                    x_stride : x_stride + self.pool_size,
                    :,
                ]
                slice_x_reshape = slice_X.reshape(
                    (batch_size, self.pool_size * self.pool_size, channels)
                )
                maxpool_inds = np.argmax(slice_x_reshape, axis=1).flatten()
                if self.tmp_result is None or not np.array_equal(
                    self.tmp_result.shape, slice_x_reshape.shape
                ):
                    self.tmp_result = np.zeros(slice_x_reshape.shape)
                self.tmp_result[batch_inds, maxpool_inds, channels_inds] = d_out[
                    batch_inds, y, x, channels_inds
                ]
                self.result[
                    :,
                    y_stride : y_stride + self.pool_size,
                    x_stride : x_stride + self.pool_size,
                    :,
                ] = self.tmp_result.reshape(
                    (batch_size, self.pool_size, self.pool_size, channels)
                )
        return self.result
    def params(self):
        return {}

class AveragePooling:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.name = "averagepooling"
        self.out = None
        self.result = None
        self.tmp_result = None
    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        if self.out is None or not np.array_equal(
            self.out.shape, [batch_size, out_height, out_width, channels]
        ):
            self.out = np.zeros([batch_size, out_height, out_width, channels])
        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride
                self.out[:, y, x, :] = np.mean(
                    X[
                        :,
                        y_stride : y_stride + self.pool_size,
                        x_stride : x_stride + self.pool_size,
                        :,
                    ],
                    axis=(1, 2),
                )
        return self.out
    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        if self.result is None or not np.array_equal(self.result.shape, self.X.shape):
            self.result = np.zeros(self.X.shape)
        batch_inds, channels_inds = np.repeat(range(batch_size), channels), np.tile(
            range(channels), batch_size
        )
        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride
                slice_X = self.X[
                    :,
                    y_stride : y_stride + self.pool_size,
                    x_stride : x_stride + self.pool_size,
                    :,
                ]
                slice_x_reshape = slice_X.reshape(
                    (batch_size, self.pool_size * self.pool_size, channels)
                )
                maxpool_inds = np.argmax(slice_x_reshape, axis=1).flatten()
                if self.tmp_result is None or not np.array_equal(
                    self.tmp_result.shape, slice_x_reshape.shape
                ):
                    self.tmp_result = np.zeros(slice_x_reshape.shape)
                self.tmp_result[batch_inds, maxpool_inds, channels_inds] = d_out[
                    batch_inds, y, x, channels_inds
                ]
                self.result[
                    :,
                    y_stride : y_stride + self.pool_size,
                    x_stride : x_stride + self.pool_size,
                    :,
                ] = self.tmp_result.reshape(
                    (batch_size, self.pool_size, self.pool_size, channels)
                )
        return self.result
    def params(self):
        return {}

class Flattener:
    def __init__(self):
        self.X_shape = None
        self.name = "flattener"
    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = batch_size, height, width, channels
        return X.reshape(batch_size, -1)
    def backward(self, d_out):
        return d_out.reshape(self.X_shape)
    def params(self):
        return {}
