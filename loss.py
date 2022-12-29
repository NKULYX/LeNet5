import numpy as np


def softmax_loss(y_pred, y):
    # y_pred: (N, C)
    # y: (N, 1)
    N = y_pred.shape[0]
    ex = np.exp(y_pred)
    sumx = np.sum(ex, axis=1)
    loss = np.mean(np.log(sumx)-y_pred[range(N), list(y)])
    grad = ex/sumx.reshape(N, 1)
    grad[range(N), list(y)] -= 1
    grad /= N
    acc = np.mean(np.argmax(ex/sumx.reshape(N, 1), axis=1) == y.reshape(1, y.shape[0]))
    return loss, grad, acc
