import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


def sigmoid(z):
    """
    Maps an input to an output of a value between 0 and 1.
    :param z:
    :return: float, [0,1]
    """
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    sb.set_theme()
    X = np.linspace(-8, 8, 1000)
    Y = sigmoid(X)

    plt.plot(X, Y, label=r"$\sigma(t) = \frac{1}{x}$")

    plt.axhline(y=0, linestyle="--")
    plt.axhline(y=0.5, linestyle="--")
    plt.axhline(y=1, linestyle="--")

    plt.axvline(color="black")

    plt.show()
