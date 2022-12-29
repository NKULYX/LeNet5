import numpy as np


class Layer(object):
    def __init__(self):
        pass

    def forward(self, X):
        pass

    def backward(self, back_grad):
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

    def backward(self, back_grad):
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

    def forward(self, X):
        h_size = self.pool_size[0]
        w_size = self.pool_size[1]
        N, C, H, W = X.shape
        output_H = H // h_size
        output_W = W // w_size
        self.input = X.copy()
        self.output = np.zeros((N, C, output_H, output_W))
        for h in range(output_H):
            for w in range(output_W):
                self.output[:, :, h, w] = np.max(X[:, :, h*h_size:(h+1)*h_size, w*w_size:(w+1)*w_size], axis=(2, 3))
        return self.output

    def backward(self, back_grad):
        h_size = self.pool_size[0]
        w_size = self.pool_size[1]
        N, C, H, W = self.input.shape
        output_H = H // h_size
        output_W = W // w_size
        grad = np.zeros_like(self.input)
        for h in range(output_H):
            for w in range(output_W):
                tmp_x = self.input[:, :, h*h_size:(h+1)*h_size, w*w_size:(w+1)*w_size].reshape((N, C, -1))
                mask = np.zeros((N, C, h_size*w_size))
                mask[np.arange(N)[:, None], np.arange(C)[None, :], np.argmax(tmp_x, axis=2)] = 1
                grad[:, :, h*h_size:(h+1)*h_size, w*w_size:(w+1)*w_size] = mask.reshape((N, C, h_size, w_size)) * back_grad[:, :, h, w][:, :, None, None]
        return grad


class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = None
        self.output = None
        self.input_size = input_size
        self.output_size = output_size
        self.W = {'value': np.random.normal(scale=1e-3, size=(input_size, output_size)),
                  'grad': np.zeros((input_size, output_size))}
        self.b = {'value': np.zeros(output_size),
                  'grad': np.zeros(output_size)}

    def forward(self, X):
        self.input = X.copy()
        self.output = np.dot(X, self.W['value']) + self.b['value']
        return self.output

    def backward(self, back_grad):
        grad = np.dot(back_grad, self.W['value'].T)
        self.W['grad'] = np.dot(self.input.T, back_grad)
        self.b['grad'] = np.sum(back_grad, axis=0)
        return grad


class Conv(Layer):
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0):
        super().__init__()
        self.input = None
        self.output = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = {'value': np.random.normal(scale=1e-3, size=(out_channels, in_channels, filter_size, filter_size)),
                  'grad': np.zeros((out_channels, in_channels, filter_size, filter_size))}
        self.b = {'value': np.zeros(out_channels),
                  'grad': np.zeros(out_channels)}
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        self.input = X.copy()
        N, C, H, W = X.shape
        padding_x = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        output_H = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        output_W = (W + 2 * self.padding - self.filter_size) // self.stride + 1
        self.output = np.zeros((N, self.out_channels, output_H, output_W))
        for h in range(output_H):
            for w in range(output_W):
                tmp_x = padding_x[:, :, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size].reshape((N, 1, self.in_channels, self.filter_size, self.filter_size))
                tmp_W = self.W['value'].reshape((1, self.out_channels, self.in_channels, self.filter_size, self.filter_size))
                self.output[:, :, h, w] = np.sum(tmp_x * tmp_W, axis=(2, 3, 4)) + self.b['value']
        return self.output

    def backward(self, back_grad):
        N, C, H, W = self.input.shape
        padding_x = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        output_H = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        output_W = (W + 2 * self.padding - self.filter_size) // self.stride + 1
        self.W['grad'] = np.zeros((self.out_channels, self.in_channels, self.filter_size, self.filter_size))
        self.b['grad'] = np.zeros(self.out_channels)
        grad = np.zeros_like(padding_x)
        for h in range(output_H):
            for w in range(output_W):
                tmp_back_grad = back_grad[:, :, h, w].reshape((N, 1, 1, 1, self.out_channels))
                tmp_W = self.W['value'].transpose((1, 2, 3, 0)).reshape((1, self.in_channels, self.filter_size, self.filter_size, self.out_channels))
                grad[:, :, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size] += np.sum(tmp_back_grad * tmp_W, axis=4)
                tmp_back_grad = back_grad[:, :, h, w].T.reshape((self.out_channels, 1, 1, 1, N))
                tmp_x = padding_x[:, :, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size].transpose((1, 2, 3, 0))
                self.W['grad'] += np.sum(tmp_back_grad * tmp_x, axis=4)
                self.b['grad'] += np.sum(back_grad[:, :, h, w], axis=0)
        grad = grad[:, :, self.padding:self.padding + H, self.padding:self.padding + W]
        return grad
