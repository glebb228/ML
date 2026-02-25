import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)

    values = np.unique(feature_vector)
    if len(values) <= 1:
        return np.array([]), np.array([]), None, None

    thresholds = (values[:-1] + values[1:]) / 2
    n = len(target_vector)

    left_mask = feature_vector[:, None] < thresholds[None, :]
    n_left = left_mask.sum(axis=0)
    n_right = n - n_left

    valid = (n_left > 0) & (n_right > 0)
    if not np.any(valid):
        return thresholds, np.full(len(thresholds), -np.inf), None, None

    left_class1 = (target_vector[:, None] * left_mask).sum(axis=0)
    right_class1 = target_vector.sum() - left_class1

    p_left = np.divide(left_class1, n_left, where=n_left > 0, out=np.zeros_like(left_class1, dtype=float))
    p_right = np.divide(right_class1, n_right, where=n_right > 0, out=np.zeros_like(right_class1, dtype=float))

    H_left = 1 - p_left**2 - (1 - p_left)**2
    H_right = 1 - p_right**2 - (1 - p_right)**2

    ginis = -(n_left / n) * H_left - (n_right / n) * H_right
    ginis[~valid] = -np.inf

    best_index = np.argmax(ginis)
    return thresholds, ginis, thresholds[best_index], ginis[best_index]


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

        self._tree = {}

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]

            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])

                ratio = {}
                for key, cnt in counts.items():
                    ratio[key] = clicks.get(key, 0) / cnt

                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = dict(zip(sorted_categories, range(len(sorted_categories))))
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])

            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue

            current_split = feature_vector < threshold

            if self._min_samples_leaf is not None:
                if np.sum(current_split) < self._min_samples_leaf or np.sum(~current_split) < self._min_samples_leaf:
                    continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = current_split

                if feature_type == "real":
                    threshold_best = threshold
                else:
                    threshold_best = [k for k, v in categories_map.items() if v < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(np.array(X), np.array(y), self._tree, depth=0)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)