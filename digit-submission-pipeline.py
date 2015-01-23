'''
Digit recognition for the Kaggle challenge http://www.kaggle.com/c/digit-recognizer
This is the script I used to make my submission.
'''

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Loading the dataset in memory
data = pd.read_csv("train.csv")
data = data.values

test = pd.read_csv("test.csv")
test = test.values

# Separating the labels from the training set
train = data[:, 1:]
labels = data[:, :1]

# Create a classifier, KNN

estimators = [('reduce_dim', PCA(n_components = 100)), ('Knn', KNeighborsClassifier(weights = 'distance', n_neighbors=5, p=3))]
knn = Pipeline(estimators)

# The learning is done on the first half of the dataset
knn.fit(train, labels)

# Now predict the value of the digit on the second half:
predicted = knn.predict(test)

predicted = pd.DataFrame(predicted)

predicted['ImageId'] = predicted.index + 1
predicted = predicted[['ImageId', 0]]
predicted.columns = ['ImageId', 'Label']

predicted.to_csv('pred.csv', index=False)
