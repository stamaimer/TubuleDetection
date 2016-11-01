# -*- coding: utf-8 -*-

"""

    stamaimer 10/31/16

"""

import sys
import math

import numpy as np

from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score

np.set_printoptions(threshold=np.nan)

client = MongoClient()

db = client.TubuleDetection.eigenvectors

divider = int(sys.argv[1])

positive = db.find({"label": 1}, {"eigenvector": 1, "label": 1, "_id": 0})
negative = db.find({"label": 0}, {"eigenvector": 1, "label": 1, "_id": 0})

positive = list(positive)
negative = list(negative)[:1170]

for ele in positive:

    ele["eigenvector"].pop(4)

    for i in xrange(len(ele["eigenvector"])):

        if math.isnan(ele["eigenvector"][i]):

            ele["eigenvector"][i] = 0.0

for ele in negative:

    ele["eigenvector"].pop(4)

    for i in xrange(len(ele["eigenvector"])):

        if math.isnan(ele["eigenvector"][i]):

            ele["eigenvector"][i] = 0.0


def chunks(l, n):

    for i in xrange(0, len(l), n):

        yield l[i:i + n]


positive_split_data = list(chunks(positive, len(positive) / divider))

negative_split_data = list(chunks(negative, len(negative) / divider))

accuracys_for_all_test_data = []
accuracys_for_positive_data = []
accuracys_for_negative_data = []

for i in xrange(divider):

    positive_test_data = [item["eigenvector"] for item in positive_split_data[i]]
    negative_test_data = [item["eigenvector"] for item in negative_split_data[i]]

    test_data = positive_test_data + negative_test_data

    positive_test_target = [int(item["label"]) for item in positive_split_data[i]]
    negative_test_target = [int(item["label"]) for item in negative_split_data[i]]

    test_target = positive_test_target + negative_test_target

    positive_train_data = [item["eigenvector"] for data in positive_split_data[:i] + positive_split_data[i + 1:] for item in data]
    negative_train_data = [item["eigenvector"] for data in negative_split_data[:i] + negative_split_data[i + 1:] for item in data]

    train_data = positive_train_data + negative_train_data

    positive_train_target = [int(item["label"]) for data in positive_split_data[:i] + positive_split_data[i + 1:] for item in data]
    negative_train_target = [int(item["label"]) for data in negative_split_data[:i] + negative_split_data[i + 1:] for item in data]

    train_target = positive_train_target + negative_train_target

    classifier = RandomForestClassifier(n_estimators=1000)

    classifier.fit(train_data, train_target)

    accuracy_for_all_test_data = accuracy_score(test_target, classifier.predict(test_data))
    accuracys_for_all_test_data.append(accuracy_for_all_test_data)
    accuracy_for_positive_data = accuracy_score(positive_test_target, classifier.predict(positive_test_data))
    accuracys_for_positive_data.append(accuracy_for_positive_data)
    accuracy_for_negative_data = accuracy_score(negative_test_target, classifier.predict(negative_test_data))
    accuracys_for_negative_data.append(accuracy_for_negative_data)

    print "========================================="
    print "Accuracy for all test data", accuracy_for_all_test_data
    print "Accuracy for positive data", accuracy_for_positive_data
    print "Accuracy for negative data", accuracy_for_negative_data

print "========================================="
print "Average accuracy for all test data of {0} cross validation is {1}".format(divider, sum(accuracys_for_all_test_data) / divider)
print "Average accuracy for positive data of {0} cross validation is {1}".format(divider, sum(accuracys_for_positive_data) / divider)
print "Average accuracy for negative data of {0} cross validation is {1}".format(divider, sum(accuracys_for_negative_data) / divider)