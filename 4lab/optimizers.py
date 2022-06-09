import numpy as np


class Adam:
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.t = 1
        self.v = 0
        self.s = 0

    def update(self, weights, grad, learning_rate):
        self.v = self.beta_1 * self.v + (1 - self.beta_1) * grad
        self.s = self.beta_2 * self.s + (1 - self.beta_2) * np.square(grad)

        v_bias_corr = self.v / (1 - self.beta_1**self.t)
        s_bias_corr = self.s / (1 - self.beta_2**self.t)
        weights -= learning_rate * v_bias_corr / (np.sqrt(s_bias_corr) + self.epsilon)
        self.t += 1

        return weights
