import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def split_data_set(n_rows=1000, train_test_split=0.1):
    """
    Split data-set into training and testing set.
    :return: [x_train, y_train, x_test, y_test]
    """
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, r"data_set\train\advertising.csv")

    df = pd.read_csv(file_path, nrows=n_rows)

    x = df.drop(['click', 'ad_topic_line'], axis=1).values
    y = df['click'].values

    n_train = int(n_rows * (1 - train_test_split))

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(x_train)

    x_train_enc = enc.fit_transform(x_train)
    x_test_enc = enc.transform(x_test)

    return x_train_enc.toarray(), y_train, x_test_enc.toarray(), y_test
