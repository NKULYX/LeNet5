import numpy as np


class Layer(object):
    def __init__(self):
        pass

    def forward(self, X):
        pass

    def backward(self, grad, lr=1e-5):
        pass


class ReLu(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X.copy()
        self.output = np.maximum(0, X)
        return self.output

    def backward(self, back_grad, lr=1e-5):
        grad = self.input > 0
        return back_grad * grad


class MaxPool(Layer):
    def __init__(self, pool_size=None):
        super().__init__()
        if pool_size is None:
            pool_size = [2, 2]
        self.input = None
        self.output = None
        self.pool_size = pool_size
        self.mask = None

    def forward(self, X):
        self.input = X.copy()
        n, c, h, w = X.shape
        h_out = h // self.pool_size[0]
        w_out = w // self.pool_size[1]
        self.output = np.zeros((n, c, h_out, w_out))
        self.mask = np.zeros_like(self.input)
        h_stride = self.pool_size[0]
        w_stride = self.pool_size[1]
        for i in range(h_out):
            for j in range(w_out):
                max_value = np.max(X[:, :, h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1)], axis=(2, 3))
                self.output[:, :, i, j] = max_value
                for p in range(n):
                    for q in range(c):
                        self.mask[p, q, h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1)] = X[p, q, h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1)] == max_value[p][q]
                        if np.allclose(self.mask[p, q, h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1)], np.ones_like(self.mask[p, q, h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1)])):
                            self.mask[p, q, h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1)] = 0
                            self.mask[p, q, h_stride*i, w_stride*j] = 1
        return self.output

    def backward(self, back_grad, lr=1e-5):
        n, c, h, w = self.input.shape
        h_out = h // self.pool_size[0]
        w_out = w // self.pool_size[1]
        h_stride = self.pool_size[0]
        w_stride = self.pool_size[1]
        grad = np.zeros_like(self.input)
        for i in range(h_out):
            for j in range(w_out):
                grad[:, :, h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1)] += back_grad[:, :, i:i+1, j:j+1]
        return grad * self.mask


class FullyConnect(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = None
        self.output = None
        self.W = np.random.normal(0, np.sqrt(2 / input_size), (input_size, output_size))
        self.b = np.random.randn(output_size)

    def forward(self, X):
        self.input = X.copy()
        self.output = np.dot(X.reshape(X.shape[0], -1), self.W) + self.b
        return self.output

    def backward(self, back_grad, lr=1e-5):
        grad = np.dot(back_grad, self.W.T).reshape(self.input.shape)
        dW = np.dot(self.input.reshape(self.input.shape[0], -1).T, back_grad)
        db = np.sum(back_grad, axis=0)
        adaptive_lr = lr / np.abs(np.max(dW) - np.min(dW))
        self.W -= adaptive_lr * dW
        adaptive_lr = lr / np.abs(np.max(db) - np.min(db))
        self.b -= adaptive_lr * db
        return grad


class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X.copy()
        self.output = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output /= np.sum(self.output, axis=1, keepdims=True)
        return self.output

    def backward(self, back_grad, lr=1e-5):
        softmax_grad = []
        for i in range(back_grad.shape[0]):
            softmax_grad.append(np.dot((np.diag(self.output[i]) - np.outer(self.output[i], self.output[i])), back_grad[i].T))
        return np.array(softmax_grad)


class Conv(Layer):
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        # 初始化权重
        self.weights = np.random.normal(0, np.sqrt(2 / in_channels), (out_channels, in_channels, filter_size, filter_size))
        self.biases = np.random.randn(out_channels)

        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X.copy()
        padding_x = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        N, C, H, W = X.shape
        output_H = (H - self.filter_size + 2 * self.padding) // self.stride + 1
        output_W = (W - self.filter_size + 2 * self.padding) // self.stride + 1
        self.output = np.zeros((N, self.out_channels, output_H, output_W))

        for i in range(output_H):
            for j in range(output_W):
                tmp_x = padding_x[:, :, i * self.stride:i * self.stride + self.filter_size, j * self.stride:j * self.stride + self.filter_size].reshape(N, 1, C, self.filter_size, self.filter_size)
                tmp_w = self.weights.reshape(1, self.out_channels, self.in_channels, self.filter_size, self.filter_size)
                self.output[:, :, i, j] = np.sum(tmp_x * tmp_w, axis=(-3, -2, -1))
                self.output[:, :, i, j] += self.biases.reshape(1, self.out_channels)

        return self.output

    def backward(self, back_grad, lr=1e-5):
        N, C, H, W = self.input.shape
        output_H = self.output.shape[2]
        output_W = self.output.shape[3]
        padding_x = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        grad = np.zeros_like(self.input)
        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.biases)

        for i in range(output_H):
            for j in range(output_W):
                tmp_out = back_grad[:, :, i, j].reshape(N, 1, 1, 1, self.out_channels)
                tmp_w = self.weights.transpose(1, 2, 3, 0).reshape(1, self.in_channels, self.filter_size, self.filter_size, self.out_channels)
                grad[:, :, i * self.stride:i * self.stride + self.filter_size, j * self.stride:j * self.stride + self.filter_size] += np.sum(tmp_w * tmp_out, axis=-1)
                tmp_out = back_grad[:, :, i, j].reshape(self.out_channels, 1, 1, 1, N)
                tmp_in = padding_x[:, :, i * self.stride:i * self.stride + self.filter_size, j * self.stride:j * self.stride + self.filter_size].transpose(1, 2, 3, 0).reshape(1, self.in_channels, self.filter_size, self.filter_size, N)
                dW += np.sum(tmp_out * tmp_in, axis=-1)
                db += np.sum(back_grad[:, :, i, j], axis=0)

        # 自适应学习率
        adaptive_lr = lr / np.abs(np.max(dW) - np.min(dW))
        self.weights -= adaptive_lr * dW
        adaptive_lr = lr / np.abs(np.max(db) - np.min(db))
        self.biases -= lr * db
        return grad[:, :, self.padding:self.padding + H, self.padding:self.padding + W]

