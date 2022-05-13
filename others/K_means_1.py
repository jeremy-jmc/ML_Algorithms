from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from operator import itemgetter
import matplotlib.pyplot as plt


def euclidean(x1, x2):
    return distance.euclidean(x1, x2)


def chebychev(x1, x2):
    return distance.chebyshev(x1, x2)


iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])

X = df[['sepal length (cm)', 'petal length (cm)', 'target']]
# X = df.loc[:, df.columns in ['sepal_length', 'petal_length']]
y = df.loc[:, df.columns == 'target']


class K_means:
    def __init__(self, X, y, k, distance=euclidean):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=0.30, random_state=42)
        self.k = k
        self.distance = distance

        self.X = X
        self.y = y

        self.centroids_X = self.X_train.sample(n=k, random_state=42)
        self.centroids_index = self.centroids_X.index.to_list()
        self.centroids_y = self.y_train.filter(items=self.centroids_index, axis=0)
        self.centroids_X['elements'] = 1

        self.X_train.drop(self.centroids_index, inplace=True, axis=0)
        self.y_train.drop(self.centroids_index, inplace=True, axis=0)

    def get_index_of_nearest_centroid(self, row):
        min_distance = float('inf')
        index = 0

        for i in range(len(self.centroids_X.index)):
            current_centroid = self.centroids_X.iloc[i]
            current_distance = self.distance(current_centroid.tolist()[:-1], row.tolist())

            if current_distance < min_distance:
                index = i
                min_distance = current_distance
        return index

    def get_label_of_centroid(self, index):
        return self.centroids_y.iloc[index]['target']

    def update_centroid(self, centroid_index, row):

        values_centroid = self.centroids_X.iloc[centroid_index]
        values = values_centroid.tolist()

        values_row = row.tolist()

        elementos = values[-1]
        new_size = elementos + 1

        for i in range(len(values) - 1):
            values[i] *= elementos
            values[i] += values_row[i]
            values[i] /= new_size

        values[-1] += 1

        self.centroids_X.iloc[centroid_index] = values

    def execute(self):
        old_values_centroids = self.centroids_X.copy()

        for i in range(len(self.X_train.index)):
            row = self.X_train.iloc[i]
            centroid_index = self.get_index_of_nearest_centroid(row)
            self.update_centroid(centroid_index, row)

        while not old_values_centroids.equals(self.centroids_X):
            for i in range(len(self.X_train.index)):
                row = self.X_train.iloc[i]
                centroid_index = self.get_index_of_nearest_centroid(row)
                self.update_centroid(centroid_index, row)
            old_values_centroids = self.centroids_X.copy()

    def score(self):
        scores = {}
        colors = ['blue', 'green', 'red']
        markers = ['s', 'o', 'x']

        for index in range(len(self.centroids_X)):
            scores[index] = 0

        for index in range(len(self.X)):
            row = self.X.iloc[index]
            centroid_index = self.get_index_of_nearest_centroid(row)
            scores[centroid_index] += 1
            plt.plot(row['sepal length (cm)'], row['petal length (cm)'],
                     c=colors[int(row['target'])],
                     marker=markers[centroid_index])

        for i in range(len(self.centroids_X)):
            row = self.centroids_X.iloc[i]
            plt.plot(row['sepal length (cm)'], row['petal length (cm)'],
                     c='yellow', markersize=22,
                     marker='o')
        plt.show()
        print(scores)


cl = K_means(X, y, 3, distance=chebychev)
cl.execute()
cl.score()
