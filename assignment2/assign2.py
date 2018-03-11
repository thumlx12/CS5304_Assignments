# template for assign2.py
import numpy as np
from collections import Counter
from sklearn import preprocessing


def str2int(s, base, default=0):
    try:
        val = max(0, int(s, base))
    except:
        val = default
    finally:
        return val


def read_data(data_path, train_path, validation_path, test_path):
    train_ids = set([int(line) for line in open(train_path, 'r')])
    validation_ids = set([int(line) for line in open(validation_path, 'r')])
    test_ids = set([int(line) for line in open(test_path, 'r')])

    train_data, train_target = np.zeros((1000000, 39)), np.zeros((1000000,))
    validation_data, validation_target = np.zeros((250000, 39)), np.zeros((250000,))
    test_data, test_target = np.zeros((750000, 39)), np.zeros((750000,))

    train_cnt = 0
    validation_cnt = 0
    test_cnt = 0
    with open(data_path) as infile:
        for idx, line in enumerate(infile):
            splits = line.split('\t')
            if idx in train_ids:
                train_target[train_cnt] = int(splits[0])
                train_data[train_cnt, 0:13] = np.array([str2int(splits[i], 10) for i in range(1, 14)])
                train_data[train_cnt, 13:] = np.array([str2int(splits[i], 16) for i in range(14, 40)])
                train_cnt += 1
            elif idx in validation_ids:
                validation_target[validation_cnt] = int(splits[0])
                validation_data[validation_cnt, 0:13] = np.array([str2int(splits[i], 10) for i in range(1, 14)])
                validation_data[validation_cnt, 13:] = np.array([str2int(splits[i], 16) for i in range(14, 40)])
                validation_cnt += 1
            elif idx in test_ids:
                test_target[test_cnt] = int(splits[0])
                test_data[test_cnt, 0:13] = np.array([str2int(splits[i], 10) for i in range(1, 14)])
                test_data[test_cnt, 13:] = np.array([str2int(splits[i], 16) for i in range(14, 40)])
                test_cnt += 1

    return train_data, train_target, validation_data, validation_target, test_data, test_target


def preprocess_int_data(data, features):
    int_features = list(filter(lambda f: f < 13, features))
    return preprocessing.scale(data[:, int_features])


def getMostCommonCatVals(data, threshold=0.9):
    counter = Counter(data)
    if len(counter) <= 20:
        return list(counter.keys())
    major_vals = []
    cnt = 0
    total = float(len(data))
    for entry in counter.most_common():
        if cnt / total > threshold:
            break
        cnt += entry[1]
        major_vals.append(entry[0])
    return major_vals


def onehotTransform(data, major_vals):
    transformed = np.zeros((len(data), len(major_vals)), dtype='int')
    for i in range(len(data)):
        if data[i] in major_vals:
            transformed[i, major_vals.index(data[i])] = 1
    return transformed


def preprocess_cat_data(data, features, preprocess):
    cat_features = list(filter(lambda f: f >= 13, features))
    return np.hstack([onehotTransform(data[:, f], getMostCommonCatVals(data[:, f])) for f in cat_features])
