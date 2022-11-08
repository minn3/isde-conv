import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from conv_1d_kernels.CConvKernelMovingAverage import CConvKernelMovingAverage


def load_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    y = data[:, 0]  # all rows and first column
    x = data[:, 1:] / 255  # all rows and remaing columns
    return x, y


def plot_digit(image, shape=(28, 28)):
    plt.imshow(np.reshape(image, newshape=(28, 28)), 'gray')


x, y = load_data("../data/mnist_data.csv")

conv = CConvKernelMovingAverage(7)

plt.figure()
plot_digit(x[0])
plt.show()

image_blur = conv.kernel(x[0])

plt.figure()
plot_digit(image_blur)
plt.show()