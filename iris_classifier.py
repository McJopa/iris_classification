import numpy as np

#Import the Flower Iris dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris();

#split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0);

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train);

y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)*100
print('Accuracy: %d%%' %accuracy)
