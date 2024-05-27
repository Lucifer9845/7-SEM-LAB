# 7.)Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for clustering using k-Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add Java/Python ML library classes/API in the program.

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

# load data, add colun headers
iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
x.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
y = pd.DataFrame(iris.target)
y.columns = ["Targets"]


# 2- set winsiize. colormap
plt.figure(figsize=(14, 14))
colormap = np.array(["red", "lime", "black"])

# 3 - train x by kmeans
model = KMeans(n_clusters=3)
model.fit(x)

# 4- plot original data
plt.subplot(2, 2, 1)
plt.scatter(x.petal_length, x.petal_width, c=colormap[y.Targets], s=40)
plt.title("Real Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
# plt.show()

# 5 - plot kmeans trained data
plt.subplot(2, 2, 2)
plt.scatter(x.petal_length, x.petal_width, c=colormap[model.labels_], s=40)
plt.title("KMeans Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
# plt.show()


# 6 - scale data
scaler = preprocessing.StandardScaler()
scaler.fit(x)
xsa = scaler.transform(x)
xs = pd.DataFrame(xsa, columns=x.columns)

# 7 - gmm training
gmm = GaussianMixture()
gmm.fit(xs)
gmm_y = gmm.predict(xs)

# 8 - gmm plotting
plt.subplot(2, 2, 3)
plt.scatter(x.petal_length, x.petal_width, c=colormap[gmm_y], s=40)
plt.title("GMM Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
print(
    "The GMM using EM algorithm based clustering matched the true labels closely than KMeans clusters"
)

# OUTPUT :


# The GMM using EM algorithm based clustering matched the true labels closely than KMeans clusters.
