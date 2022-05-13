
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('./iris.csv')
print(df.head(5))

# sns.pairplot(data=df, hue='variety')

def euclidean(x1, x2):
    return distance.euclidean(x1, x2)

def chebychev(x1, x2):
    return distance.chebyshev(x1, x2)

def manhattan(x1, x2):
    return distance.cityblock(x1, x2)

# %%
class K_Means():
    def __init__(self, data, n, distance=euclidean):
        self.data = data.to_numpy()
        self.n = n  # n° centroids
        self.K = None
        self.d = distance

    def new_centroids(self, idx):
        idx_df = pd.DataFrame(idx, columns=['idx', 'cluster'])
        new_K = []
        for i in range(len(self.K)):
            # obtener indices de todos los elementos pertenecientes al cluster i
            curr_cluster = idx_df.loc[idx_df['cluster'] == i]           
            elements_cluster = np.matrix(self.data[curr_cluster['idx'].to_list(), :])
            if len(elements_cluster) == 0:
                new_K.append(self.K[i])  # empty cluster
            else:
                new_K.append(np.array(elements_cluster.mean(axis=0))[0])
        return np.array(new_K, dtype=object)

    def label(self):
        idx = np.array([
            min([
                (id, i, self.d(self.K[i], self.data[id, :])) for i in range(len(self.K))
            ], key=itemgetter(2))[:-1] for id in range(len(self.data))
        ])
        return idx  # array[(index, class), ...]

    def execute(self):
        r, c = self.data.shape
        c_idx = np.random.choice(r, self.n)
        idx = None

        new_K = self.data[c_idx, :]  # .iloc
        # print(f'initial c_idx centroids\n {c_idx}')
        i = 0
        while not (self.K == new_K).all():
            self.K = new_K
            # print(f'current_centroids \n {self.K}')
            idx = self.label()
            new_K = self.new_centroids(idx)
            i += 1
            # print(f'\tK\n{self.K} \n\tK_new\n{new_K}')
        print(f'\titerations {i}')
        return idx


# %%
n = 3
d = df.iloc[:,:-1]  # df.sample(n=5).iloc[:,:-1]
k = K_Means(d, 3)

# %%
clusters = k.execute()
clusters
df['idx'] = clusters[:,1]

# %%
print(df)

# %%
print(df['variety'].unique())
df = df.loc[df['variety'].isin(['Setosa', 'Versicolor', 'Virginica'])].replace({'variety':{'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}})

# %%
cm = confusion_matrix(df['idx'], df['variety'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# %%
# 'variety':{'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
cm_normalize = normalize(cm, norm='l1')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalize)
disp.plot()

# %%
# HACER AVI



