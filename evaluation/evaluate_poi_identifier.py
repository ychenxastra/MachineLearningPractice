#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn import tree

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### your code goes here 

features_train, features_test, labels_train, labels_test = train_test_split(features,
                labels,test_size=0.3,random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
clf.score(features_test, labels_test)

pred = clf.predict(features_test)

# sum(pred)
print("How many POIs are predicted for the test set for your POI identifier??",
    len(features_test))
print("How many people total are in your test set?", len(pred))
print(1.0 - 5.0/29)

from sklearn.metrics import *
print("What’s the precision?", precision_score(labels_test, pred))
print("What’s the recall?",recall_score(labels_test, pred))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print("precision:", precision_score(true_labels, predictions))
print("recall:", recall_score(true_labels, predictions))
print("f1_score", f1_score(true_labels, predictions))