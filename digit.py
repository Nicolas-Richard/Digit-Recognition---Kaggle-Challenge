'''
Digit recognition for the Kaggle challenge http://www.kaggle.com/c/digit-recognizer
This is the script I used to explore the data and check how good the model could using
cross validation.
'''

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Loading the dataset in memory
data = pd.read_csv("train.csv")
data = data.values

data = data[:100]

n_samples = data.shape[0]

# Separating the labels from the training set
train = data[:, 1:]
labels = data[:, :1]

# Create a classifier, KNN
knn = KNeighborsClassifier(weights = 'distance', n_neighbors=10, p=3)

# The learning is done on the first half of the dataset
knn.fit(train[:n_samples / 2], labels[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = labels[n_samples / 2:]
predicted = knn.predict(train[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (knn, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
