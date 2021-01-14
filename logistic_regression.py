import numpy as np


def sigmoid_activation_function(z):
    """
    Maps an input to an output of a value between 0 and 1.
    :param z:
    :return: float, [0,1]
    """
    val = 1 / (1 + np.exp(-z))
    return val


class LogisticRegression:
    """
    Logistic regression is a probabilistic classifier, similar to the Naive Bayes classifier.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, fit_intercept=False, optimizer='gd', verbose=0):
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter
        self.__fit_intercept = fit_intercept
        self.__optimizer = optimizer
        self.__weights = None
        self.__verbose = verbose

    def __compute_prediction(self, x):
        """
        Compute the prediction y_hat based on current weights
        :param x:
        :return: numpy.ndarray, y_hat of x under weights
        """
        z = np.dot(x, self.__weights)
        predictions = sigmoid_activation_function(z)
        return predictions

    def __update_weights_grad_desc(self, x_train, y_train):
        """
        Update weights by one step
        :param x_train:
        :param y_train:
        :return:
        """

        predictions = self.__compute_prediction(x_train)
        weights_delta = np.dot(x_train.T, y_train - predictions)

        m = y_train.shape[0]
        self.__weights += self.__learning_rate / float(m) * weights_delta

    def __update_weights_stoc_gd(self, x_train, y_train):
        """
        One weight update iteration: moving weights by one step based on each individual sample.
        :param x_train:
        :param y_train:
        :return: numpy.ndarray, update weights
        """

        for x_each, y_each in zip(x_train, y_train):
            prediction = self.__compute_prediction(x_each)
            weights_delta = x_each.T * (y_each - prediction)
            self.__weights += self.__learning_rate * weights_delta

    def __compute_cost(self, x, y):
        """
        Compute the cost J(w)
        :param x:
        :param y:
        :return:
        """

        predictions = self.__compute_prediction(x)
        cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))

        return cost

    def fit(self, x_train, y_train):
        """
        Train a logistic regression model
        :param x_train:
        :param y_train:
        :return:
        """

        optimizers_functions = {'gd': self.__update_weights_grad_desc, 'sgd': self.__update_weights_stoc_gd}

        if self.__fit_intercept:
            intercept = np.ones((x_train.shape[0], 1))
            x_train = np.hstack((intercept, x_train))

        self.__weights = np.zeros(x_train.shape[1])

        for iteration in range(self.__max_iter):
            optimizers_functions[self.__optimizer](x_train, y_train)

            if self.__verbose == 1 and iteration % 100 == 0:
                print("Iteration: {}, training loss: {}".format(iteration, self.__compute_cost(x_train, y_train)))

    def predict(self, x):
        if x.shape[1] == self.__weights.shape[0] - 1:
            intercept = np.ones((x.shape[0], 1))
            x = np.hstack((intercept, x))

        return self.__compute_prediction(x)
