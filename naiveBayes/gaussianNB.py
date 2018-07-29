# First test program, supervised classification
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Features
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Labels
Y = np.array([1, 1, 1, 2, 2, 2])

# Classifier
clf = GaussianNB()

clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
