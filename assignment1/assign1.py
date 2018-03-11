import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_rcv1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def load_labels(path_to_labels):
    labels = pd.read_csv(path_to_labels, names=['label'], dtype=np.int32)
    return labels['label'].tolist()


def load_training_data():
    data = fetch_rcv1(subset='train')
    return data.data, data.target.toarray(), data.sample_id


def load_validation_data(path_to_ids):
    data = fetch_rcv1(subset='test')
    ids = pd.read_csv(path_to_ids, names=['id'], dtype=np.int32)
    mask = np.isin(data.sample_id, ids['id'])
    validation_data = data.data[mask]
    validation_target = data.target[mask].toarray()
    validation_ids = data.sample_id[mask]
    return validation_data, validation_target, validation_ids


def findBestKNN(X, Y, label):
    max_ave_accuracy = 0
    best_k = 0
    for k in range(2, 40):
        model = KNeighborsClassifier(n_neighbors=k)
        ave_accuracy = np.mean(cross_val_score(model, X, Y[:, label], cv=5))
        if max_ave_accuracy < ave_accuracy:
            max_ave_accuracy = ave_accuracy
            best_k = k
    print("max average accuracy for label", label, "is:", max_ave_accuracy)
    return [max_ave_accuracy, best_k]


class CS5304BaseClassifier(object):
    def __init__(self):
        pass

    def train(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class CS5304KNNClassifier(CS5304BaseClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    def __init__(self, n_neighbors=5):
        super(CS5304KNNClassifier, self).__init__()
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class CS5304NBClassifier(CS5304BaseClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
    """

    def __init__(self):
        super(CS5304NBClassifier, self).__init__()
        self.model = BernoulliNB()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class CS5304KMeansClassifier(CS5304BaseClassifier):
    def __init__(self, n_clusters=2, init='k-means++', n_init=10):
        super(CS5304KMeansClassifier, self).__init__()
        self.n_clusters = n_clusters
        self.n_init = n_init

    def train(self, X, y):
        self.centroids = self.getCentroids(X, y)
        self.model = KMeans(n_clusters=self.n_clusters, init=self.centroids, n_init=self.n_init)
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def getCentroids(self, X, y):
        trueSet = [i for i, j in enumerate(y) if j == 1]
        falseSet = [i for i, j in enumerate(y) if j == 0]
        trueCentroid = X[trueSet].mean(axis=0)
        falseCentroid = X[falseSet].mean(axis=0)
        return np.concatenate((trueCentroid, falseCentroid), axis=0)


if __name__ == '__main__':
    # This is an example of loading the training and validation data. You may use this snippet
    # when completing the exercises for the assignment.

    # This file contains working codes for (1a),(1b),(2a)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_labels", default="labels.txt")
    parser.add_argument("--path_to_ids", default="validation.txt")
    options = parser.parse_args()

    labels = load_labels(options.path_to_labels)
    train_data, train_target, _ = load_training_data()
    eval_data, eval_target, _ = load_validation_data(options.path_to_ids)
