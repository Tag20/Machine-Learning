# -*- coding: utf-8 -*-
"""KMeansClustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LIyN2wP3T5486u1r8_wu6N_rNkZC79AU
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

uploaded = files.upload()

data = pd.read_csv('Live.csv',encoding='unicode_escape')

data.describe

data.shape

data.head()

data.isnull().sum()

data.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)

data.info()

data.describe()

data.drop(['ï»¿status_id', 'status_published'], axis=1, inplace=True)

data.info()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(Xdata)

kmeans.cluster_centers_

labels = kmeans.labels_
correct_labels = sum(ydata == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, ydata.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(ydata.size)))

from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(Xdata)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

# 2 Cluters

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2,random_state=0)

kmeans.fit(Xdata)

labels = kmeans.labels_

# check how many of the samples were correctly labeled

correct_labels = sum(ydata == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, ydata.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(ydata.size)))

# 3 Clusters

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(Xdata)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(ydata == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, ydata.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(ydata.size)))

#4 Clusters

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(Xdata)
labels = kmeans.labels_
correct_labels = sum(ydata == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, ydata.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(ydata.size)))

kmeans.cluster_centers_
all_labels = kmeans.labels_

data.isna().sum()

predictors = ['num_reactions', 'num_reactions', 'num_shares', 'num_likes', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']

kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(data[predictors])

kmeans.cluster_centers_
all_labels = kmeans.labels_

fig,ax = plt.subplots(1,1, figsize = (10, 5))
ax.scatter(data.iloc[:,0], data.iloc[:,1], s = 10, c=kmeans.labels_)
ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s=20, c='r')
ax.set_title('kmeans')

