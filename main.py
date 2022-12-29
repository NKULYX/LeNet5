import numpy as np
import dataloader
from model import LeNet
from optimizer import Adam
from loss import softmax_loss
import tqdm
import matplotlib.pyplot as plt

batch_size = 256
epochs = 10
lr = 1e-3

data = dataloader.get_mnist_data()
model = LeNet()
optimizer = Adam(model.get_params(), lr)

history_loss = []
history_acc = []


def train():
    for k, v in list(data.items()):
        print(f"{k}: {v.shape}")
    best_acc = 0
    best_weight = None
    for e in range(epochs):
        # add tqdm
        pbar = tqdm.tqdm(range(0, int(data['X_train'].shape[0]/batch_size)), ncols=150)
        for i in pbar:
            X, y = dataloader.get_batch(data["X_train"], data["y_train"], batch_size)
            y_pred = model.forward(X)
            loss, grad, acc = softmax_loss(y_pred, y)
            history_loss.append(loss)
            history_acc.append(acc)
            model.backward(grad)
            optimizer.step()
            pbar.set_description(f"Epoch: {e+1}/{epochs}")
            pbar.set_postfix(loss=loss, acc=acc)

        val_X = data["X_val"]
        val_y = data["y_val"]
        y_pred = model.forward(val_X)
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.mean(y_pred == val_y.reshape(1, val_y.shape[0]))
        if acc > best_acc:
            best_acc = acc
            best_weight = model.get_params()
        pbar.set_postfix(val_acc=acc)
    return best_weight


def test(best_weight):
    X = data["X_test"]
    y = data["y_test"]
    model.set_params(best_weight)
    y_pred = model.forward(X)
    y_pred = np.argmax(y_pred, axis=1)
    acc = np.mean(y_pred == y.reshape(1, y.shape[0]))
    print(f"Test Accuracy: {acc}")


def plot_result():
    # 分别绘制 loss 和 accuracy 曲线 并保存图片
    plt.figure()
    plt.plot(history_loss)
    plt.title("history loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('loss.png')
    plt.figure()
    plt.plot(history_acc)
    plt.title("history acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.savefig('acc.png')


def main():
    best_weight = train()
    test(best_weight)
    plot_result()


if __name__ == '__main__':
    main()
