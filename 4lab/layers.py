from calendar import c
import numpy as np
import cupy as cp
from time import time

class Param:
    def __init__(self, value):
        self.value = value
        self.grad = cp.zeros_like(value)

class ReLULayer:
    def __init__(self):
        self.name = "relu"
        self.indexes = None
        pass
    def forward(self, X):
        if self.indexes is None or self.indexes.shape[0] != X.shape[0]:
            self.indexes = cp.zeros_like(X, dtype=np.bool)
        cp.less(X, 0, out=self.indexes)
        result = X
        cp.multiply(result, self.indexes, out=result)
        return result
    def backward(self, d_out):
        d_result = d_out
        cp.multiply(d_result, self.indexes, out=d_result)
        return d_result
    def params(self):
        return {}

class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * cp.random.randn(n_input, n_output))
        self.B = Param(0.001 * cp.random.randn(1, n_output))
        self.X = None
        self.name = "FC"
    def forward(self, X):
        self.X = X
        res = cp.dot(X, self.W.value) + self.B.value
        return res
    def backward(self, d_out):
        self.W.grad = cp.dot(self.X.T, d_out)
        self.B.grad = cp.array([cp.sum(d_out, axis=0)])
        gradX = cp.dot(d_out, self.W.value.T)
        return gradX
    def params(self):
        return {"W": self.W, "B": self.B}

class TanhLayer:
    def __init__(self):
        self.name = "tanh"
    def forward(self, X):
        return cp.tanh(X)
    def backward(self, d_out):
        return 1 - d_out**2
    def params(self):
        return {}

class SigmoidLayer:
    def __init__(self):
        self.name = "sigmoid"
    def forward(self, X):
        return 1 / (1 + cp.exp(-X))
    def backward(self, d_out):
        return d_out * (1 - d_out)
    def params(self):
        return {}

class RNN:
    def __init__(self, input_size, hidden_size):
        self.name = "RNN"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ax = Param(0.001 * cp.random.randn(input_size, hidden_size))
        self.W_aa = Param(0.001 * cp.random.randn(hidden_size, hidden_size))
        self.B = Param(0.001 * cp.random.randn(1, hidden_size))
    def forward(self, input_X):
        batch_size = input_X.shape[0]
        self.input_X = cp.swapaxes(input_X, 0, 1)
        self.relus = [ReLULayer() for x in input_X]
        hidden = cp.zeros((batch_size, self.hidden_size))
        self.hidden_list = [hidden]
        self.y_preds = []
        for input_x_t, relu in zip(self.input_X, self.relus):
            input_tanh = (
                cp.dot(input_x_t, self.W_ax.value)
                + cp.dot(hidden, self.W_aa.value)
                + self.B.value
            )
            hidden = relu.forward(input_tanh)
            if cp.any(cp.isnan(hidden)):
                return None
            self.hidden_list.append(hidden)
        return hidden
    def backward(self, d_out):
        for input_x_t, hidden, relu in reversed(
            list(zip(self.input_X, self.hidden_list[:-1], self.relus))
        ):
            dtanh = relu.backward(d_out)
            self.B.grad += cp.array([cp.sum(d_out, axis=0)])
            self.W_ax.grad += cp.dot(input_x_t.T, dtanh)
            self.W_aa.grad += cp.dot(hidden.T, dtanh)
            d_out = cp.dot(dtanh, self.W_aa.value.T)
        return None
    def params(self):
        return {"W_ax": self.W_ax, "W_aa": self.W_aa, "B": self.B}

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.name = "LSTM"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_f = Param(0.001 * cp.random.randn(input_size + hidden_size, hidden_size))
        self.W_i = Param(0.001 * cp.random.randn(input_size + hidden_size, hidden_size))
        self.W_c = Param(0.001 * cp.random.randn(input_size + hidden_size, hidden_size))
        self.W_o = Param(0.001 * cp.random.randn(input_size + hidden_size, hidden_size))
        self.B_f = Param(0.001 * cp.random.randn(1, hidden_size))
        self.B_i = Param(0.001 * cp.random.randn(1, hidden_size))
        self.B_c = Param(0.001 * cp.random.randn(1, hidden_size))
        self.B_o = Param(0.001 * cp.random.randn(1, hidden_size))
        self.tanh = TanhLayer()
        self.sigmoid = SigmoidLayer()
    def forward(self, input_X):
        batch_size = input_X.shape[0]
        self.input_X = cp.swapaxes(input_X, 0, 1)
        hidden = cp.zeros((batch_size, self.hidden_size))
        cell_state = cp.zeros((batch_size, self.hidden_size))
        self.hidden_list = [hidden]
        self.cell_state_list = [cell_state]
        self.y_preds = []
        self.o_t_list = []
        self.i_t_list = []
        self.f_t_list = []
        self.C_t_wave_list = []
        for input_x_t in self.input_X:
            h_x_concat = cp.concatenate((hidden, input_x_t), axis=1)
            f_t = self.sigmoid.forward(h_x_concat @ self.W_f.value + self.B_f.value)
            i_t = self.sigmoid.forward(h_x_concat @ self.W_i.value + self.B_i.value)
            C_t_wave = self.tanh.forward(h_x_concat @ self.W_c.value + self.B_c.value)
            cell_state = f_t * cell_state + i_t * C_t_wave
            o_t = self.sigmoid.forward(h_x_concat @ self.W_o.value + self.B_o.value)
            hidden = o_t * self.tanh.forward(cell_state)
            self.hidden_list.append(hidden)
            self.cell_state_list.append(cell_state)
            self.o_t_list.append(o_t)
            self.i_t_list.append(i_t)
            self.f_t_list.append(f_t)
            self.C_t_wave_list.append(C_t_wave)
        return hidden
    def backward(self, d_out):
        d_cell_state = cp.zeros_like(self.cell_state_list[0])
        for (
            input_x_t,
            hidden,
            cell_state,
            o_t,
            i_t,
            f_t,
            C_t_wave,
            prev_cell_state,
        ) in reversed(
            list(
                zip(
                    self.input_X,
                    self.hidden_list[:-1],
                    self.cell_state_list,
                    self.o_t_list,
                    self.i_t_list,
                    self.f_t_list,
                    self.C_t_wave_list,
                    self.cell_state_list[:-1],
                )
            )
        ):
            d_o_t = self.tanh.forward(cell_state) * d_out
            d_C_t = d_cell_state + d_out * o_t * (
                1 - self.tanh.forward(cell_state) ** 2
            )
            d_C_t_wave = d_C_t * i_t
            d_i_t = d_C_t * C_t_wave
            d_f_t = d_C_t * prev_cell_state
            d_f_t = f_t * (1 - f_t) * d_f_t
            d_i_t = i_t * (1 - i_t) * d_i_t
            d_o_t = o_t * (1 - o_t) * d_o_t
            d_z_t = (
                self.W_f.value @ d_f_t.T
                + self.W_i.value @ d_i_t.T
                + self.W_c.value @ d_C_t_wave.T
                + self.W_o.value @ d_o_t.T
            )
            d_out = d_z_t.T[: d_out.shape[0], : d_out.shape[1]]
            z = cp.concatenate((hidden, input_x_t), axis=1)
            self.W_f.grad += z.T @ d_f_t
            self.B_f.grad += d_f_t.sum(axis=0).reshape(1, -1)
            self.W_i.grad += z.T @ d_i_t
            self.B_i.grad += d_i_t.sum(axis=0).reshape(1, -1)
            self.W_c.grad += z.T @ d_C_t_wave
            self.B_c.grad += d_C_t_wave.sum(axis=0).reshape(1, -1)
            self.W_o.grad += z.T @ d_o_t
            self.B_o.grad += d_o_t.sum(axis=0).reshape(1, -1)
        return None
    def params(self):
        return {
            "W_f": self.W_f,
            "W_i": self.W_i,
            "W_c": self.W_c,
            "W_o": self.W_o,
            "B_f": self.B_f,
            "B_i": self.B_i,
            "B_c": self.B_c,
            "B_o": self.B_o,
        }
