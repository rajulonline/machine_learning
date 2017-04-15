import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print(cancer['target_names'])
print("Shape of Cancer data: \n{}".format(cancer.data.shape))
print("Sample counts per class: \n{}".format({n: v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Feature names: \n {}".format(cancer.feature_names))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []

#try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

from sklearn.neighbors import KNeighborsClassifier
for n_neighbors in neighbors_settings:
		#build the model		
		clf = KNeighborsClassifier(n_neighbors=n_neighbors)
		clf.fit(X_train, y_train)

		#record training accuracy
		training_accuracy.append(clf.score(X_train, y_train))

		#record generalization accuracy
		test_accuracy.append(clf.score(X_test, y_test))

import matplotlib.pyplot as plt
plt.plot(neighbors_settings, training_accuracy, label="training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))
