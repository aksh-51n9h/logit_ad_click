import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from dataset_preprocessing import split_data_set


def gini_impurity_np(labels):
    if labels.size == 0:
        return 0

    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)


def entropy_np(labels):
    if labels.size == 0:
        return 0

    counts = np.unique(labels, return_counts=True)[1]

    fractions = counts / float(len(labels))
    return -np.sum(fractions * np.log2(fractions))


def weighted_impurity(groups, criterion='gini'):
    """
    Calculated weighted impurity of children after a split
    :param groups: list of children, and a child consist a list of class labels
    :param criterion: parameter to measure the quality of a split, 'gini' for Gini Impurity or 'entropy' for Information Gain
    :return: flaot, weighted impurity
    """

    criterion_function_np = {'gini': gini_impurity_np, 'entropy': entropy_np}

    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(groups) / float(total) * criterion_function_np[criterion](group)

    return weighted_sum


def split_node(x, y, index, value):
    """
    Split data-set x, y baesd on a feature and a value
    :param x: data-set features
    :param y: data-set target
    :param index: index of the features used for splitting
    :param value: value of the feature used for splitting
    :return: left and right child(child format:  [left, right])
    """

    x_index = x[:, index]
    # if this feature is numerical
    if x[0, index].dtype.kind in ['i', 'f']:
        mask = x_index >= index

    # if this feature is categorical
    else:
        mask = x_index == value

    # split into left and right child
    left = [x[~mask, :], y[~mask]]
    right = [x[mask, :], y[mask]]

    return left, right


def get_best_split(x, y, criterion):
    """
    Obtain the best splitting point and resulting children for the data-set x, y
    :param x: data-set feature
    :param y: data-set target
    :param criterion: gini or entropy
    :return: dict { index:index of the feature,
                    value: feature, value,
                    children : left and right children }
    """

    best_index, best_value, best_score, children = None, None, 1, None

    for index in range(len(x[0])):
        for value in np.sort(np.unique(x[:, index])):
            groups = split_node(x, y, index, value)
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion)

            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups

    return {'index': best_index, 'value': best_value, 'children': children}


def get_leaf(labels):
    """
    Obtain the leaf as the majority of the labels
    :param labels:
    :return:
    """

    return np.bincount(labels).argmax()


def split(node, max_depth, min_size, depth, criterion):
    """
    Split children of a node to construct new nodes or assign them terminals
    :param node: with children info
    :param max_depth: maximal depth of the tree
    :param min_size: minimal samples required to further split a child
    :param depth: current depth of the node
    :param criterion: gini or entropy
    """

    left, right = node['children']
    del (node['children'])
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return

    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return

    # Check if the current depth exceeds the maximal depth
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])

        return

    # Check if the left child has enough samples
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        result = get_best_split(left[0], left[1], criterion)
        result_left, result_right = result['children']

        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['left'] = get_leaf(result_left[1])
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1, criterion)

    # Check if the right child has enough samples
    if left[1].size <= min_size:
        node['left'] = get_leaf(right[1])
    else:
        result = get_best_split(right[0], right[1], criterion)
        result_left, result_right = result['children']

        if result_left[1].size == 0:
            node['right'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['right'] = get_leaf(result_left[1])
        else:
            node['right'] = result
            split(node['right'], max_depth, min_size, depth + 1, criterion)


def train_tree(x_train, y_train, max_depth, min_size, criterion='gini'):
    """
    Construction of a tree starts here
    :param x_train: features
    :param y_train: target
    :param max_depth: maximal depth of the tree
    :param min_size: minimal samples required to further split a child
    :param criterion: gini or entropy
    """

    x = np.array(x_train)
    y = np.array(y_train)

    root = get_best_split(x, y, criterion)
    split(root, max_depth, min_size, 1, criterion)
    return root


def visualize_tree(CONDITION, node, depth=0):
    if isinstance(node, dict):
        if node['value'].dtype.kind in ['i', 'f']:
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['categorical']
        print('{}|- X{} {} {}'.format(depth * ' ',
                                      node['index'] + 1, condition['no'], node['value']))
        if 'left' in node:
            visualize_tree(node['left'], depth + 1)
        print('{}|- X{} {} {}'.format(depth * ' ',
                                      node['index'] + 1, condition['yes'], node['value']))
        if 'right' in node:
            visualize_tree(node['right'], depth + 1)
    else:
        print('{}[{}]'.format(depth * ' ', node))


if __name__ == "__main__":
    x_train = [['technology', 'professional'], ['fashion', 'student'], ['fashion', 'professional'],
               ['sports', 'student'],
               ['technology', 'student'], ['technology', 'retired'], ['sports', 'professional']]
    y_train = [1, 0, 0, 0, 1, 0, 1]

    tree = train_tree(x_train, y_train, 2, 2)

    CONDITION = {'numerical': {'yes': '>=', 'no': '<'}, 'categorical': {'yes': 'is', 'no': 'is_not'}}

    visualize_tree(CONDITION, tree)

    x_train, y_train, x_test, y_test = split_data_set()

    tree_sk = DecisionTreeClassifier(criterion='entropy', min_samples_split=10)
    tree_sk.fit(x_train, y_train)

    pred = tree_sk.predict(x_test)
    print("--" * 25)
    print("Decision Tree Accuracy score:",
          "Training samples: {0}, AUC on testing set: {1:.3f}".format(700, roc_auc_score(y_test, pred)), sep="\n")

    # logistic_regression = LogisticRegression(fit_intercept=True, max_iter=1000, learning_rate=0.12, verbose=0)
    #
    # logistic_regression.fit(X_train_enc.toarray(), Y_train)
    # pred = logistic_regression.predict(X_test_enc.toarray())
    #
    # print("--" * 25)
    #
    # print("Logistic Regression Accuracy score:",
    #       "Training samples: {0}, AUC on testing set: {1:.3f}".format(n_train, roc_auc_score(Y_test, pred)), sep="\n")
