from sklearn.metrics import roc_auc_score

from dataset_preprocessing import split_data_set
from logistic_regression import LogisticRegression

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = split_data_set(n_rows=1000, train_test_split=0.3)

    model = LogisticRegression(learning_rate=0.1, max_iter=1000, optimizer='gd', fit_intercept=True)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("--" * 30, "Logistic Regression (with Gradient Descent) Accuracy score:",
          "Training samples: {0}, AUC on testing set: {1:.3f}".format(len(x_train), roc_auc_score(y_test, predictions)),
          "--" * 30,
          sep="\n")

    model = LogisticRegression(learning_rate=0.1, max_iter=100, optimizer='sgd', fit_intercept=True)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("--" * 30, "Logistic Regression (with Stochastic Gradient Descent) Accuracy score:",
          "Training samples: {0}, AUC on testing set: {1:.3f}".format(len(x_train), roc_auc_score(y_test, predictions)),
          "--" * 30,
          sep="\n")
