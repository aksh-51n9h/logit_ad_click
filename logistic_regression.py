import numpy as np

"""
Logistic regression is a probabilistic classifier, similar to the Naive Bayes classifier.
"""


def sigmoid(z):
    """
    Maps an input to an output of a value between 0 and 1.
    :param z:
    :return: float, [0,1]
    """
    return 1 / (1 + np.exp(-z))


def compute_prediction(X, weights):
    """
    Compute the prediction y_hat based on current weights
    :param X:
    :param weights:
    :return: numpy.ndarray, y_hat of x under weights
    """
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return predictions


def update_weights_gd(X_train, y_train, weights, learning_rate):
    """
    Update weights by one step
    :param X_train:
    :param y_train:
    :param weights:
    :param learning_rate:
    :return: numpy.ndarray, updated weights
    """

    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)

    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta

    return weights


def compute_cost(X, y, weights):
    """
    Compute the cost J(w)
    :param X:
    :param y:
    :param weights:
    :return: float
    """

    predictions = compute_prediction(X, weights)
    cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))

    return cost


def train_logistic_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False, ):
    """
    Train a logistic regression model
    :param X_train:
    :param y_train:
    :param max_iter:
    :param learning_rate:
    :param fit_intercept:
    :return: numpy.ndarray, learned weights
    """

    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))

    weights = np.zeros(X_train.shape[1])

    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights,
                                    learning_rate)

        if iteration % 100 == 0:
            print(compute_cost(X_train, y_train, weights))

    return weights


def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))

    return compute_prediction(X, weights)
