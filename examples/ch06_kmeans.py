print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

import gaud_sty
plt.style.use('book.mplstyle')

#
# PLOT 1 - easy data, 1 iteration
#


# Random seed
np.random.seed(666) # good
#np.random.seed(3) # ?
#np.random.seed(4) # bad


# Generate sample data
centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

# Compute clustering with Means
n_iters = 1
k_means = KMeans(init='random',n_clusters=3, max_iter=n_iters, n_init=1)
k_means.fit(X)

# Plot setup
fig = plt.figure(figsize=(4, 3))
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #colors = ['cyan', 'magenta', 'yellow']
    markers = ['.','.','.']
else:
    colors = ['#000000', '#888888', '#DDDDDD']
    markers = ['.','.','.']

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# Actual plot
for k, col, mark in zip(range(n_clusters), colors, markers):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
             markerfacecolor=col, marker=mark)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_1.png')
plt.show()

#
# PLOT 2 - easy data, 2 iterations
#


# Random seed
np.random.seed(666) # good
#np.random.seed(3) # ?
#np.random.seed(4) # bad


# Generate sample data
centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

# Compute clustering with Means
n_iters = 2
k_means = KMeans(init='random',n_clusters=3, max_iter=n_iters, n_init=1)
k_means.fit(X)

# Plot setup
fig = plt.figure(figsize=(4, 3))
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #colors = ['cyan', 'magenta', 'yellow']
    markers = ['.','.','.']
else:
    colors = ['#000000', '#888888', '#DDDDDD']
    markers = ['.','.','.']

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# Actual plot
for k, col, mark in zip(range(n_clusters), colors, markers):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
             markerfacecolor=col, marker=mark)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_2.png')
plt.show()

#
# PLOT 3 - easy data, 10 iterations
#


# Random seed
np.random.seed(666) # good
#np.random.seed(3) # ?
#np.random.seed(4) # bad


# Generate sample data
centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

# Compute clustering with Means
n_iters = 10
k_means = KMeans(init='random',n_clusters=3, max_iter=n_iters, n_init=1)
k_means.fit(X)

# Plot setup
fig = plt.figure(figsize=(4, 3))
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #colors = ['cyan', 'magenta', 'yellow']
    markers = ['.','.','.']
else:
    colors = ['#000000', '#888888', '#DDDDDD']
    markers = ['.','.','.']

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# Actual plot
for k, col, mark in zip(range(n_clusters), colors, markers):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
             markerfacecolor=col, marker=mark)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_3.png')
plt.show()

#
# PLOT 4 - easy data, 1000 iterations
#


# Random seed
np.random.seed(666) # good
#np.random.seed(3) # ?
#np.random.seed(4) # bad


# Generate sample data
centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

# Compute clustering with Means
n_iters = 1000
k_means = KMeans(init='random',n_clusters=3, max_iter=n_iters, n_init=1)
k_means.fit(X)

# Plot setup
fig = plt.figure(figsize=(4, 3))
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #colors = ['cyan', 'magenta', 'yellow']
    markers = ['.','.','.']
else:
    colors = ['#000000', '#888888', '#DDDDDD']
    markers = ['.','.','.']

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# Actual plot
for k, col, mark in zip(range(n_clusters), colors, markers):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
             markerfacecolor=col, marker=mark)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_4.png')
plt.show()

#
# PLOT 5 - difficult data, 1 iteration
#


# Random seed
#np.random.seed(666) # good
#np.random.seed(3) # ?
np.random.seed(4) # bad


# Generate sample data
centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

# Compute clustering with Means
n_iters = 1
k_means = KMeans(init='random',n_clusters=3, max_iter=n_iters, n_init=1)
k_means.fit(X)

# Plot setup
fig = plt.figure(figsize=(4, 3))
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #colors = ['cyan', 'magenta', 'yellow']
    markers = ['.','.','.']
else:
    colors = ['#000000', '#888888', '#DDDDDD']
    markers = ['.','.','.']

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# Actual plot
for k, col, mark in zip(range(n_clusters), colors, markers):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
             markerfacecolor=col, marker=mark)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_5_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_5.png')
plt.show()

#
# PLOT 6 - difficult data, 1000 iterations
#


# Random seed
#np.random.seed(666) # good
#np.random.seed(3) # ?
np.random.seed(4) # bad


# Generate sample data
centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

# Compute clustering with Means
n_iters = 1000
k_means = KMeans(init='random',n_clusters=3, max_iter=n_iters, n_init=1)
k_means.fit(X)

# Plot setup
fig = plt.figure(figsize=(4, 3))
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #colors = ['cyan', 'magenta', 'yellow']
    markers = ['.','.','.']
else:
    colors = ['#000000', '#888888', '#DDDDDD']
    markers = ['.','.','.']

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# Actual plot
for k, col, mark in zip(range(n_clusters), colors, markers):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
             markerfacecolor=col, marker=mark)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_6_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_6.png')
plt.show()


#
# PLOT 7 - difficult cluster type 1
#

#
# Generate sample data: difficult

X, labels_true = make_blobs(n_samples=500, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
X = X_aniso


#
# Compute clustering with Means

n_iters = 1000
k_means = KMeans(init='random',n_clusters=3, max_iter=n_iters, n_init=1)
k_means.fit(X)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']

k_means_cluster_centers = k_means.cluster_centers_

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_7_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_7.png')
plt.show()


#
# PLOT 8 - difficult cluster type 2
#

X, labels_true = make_blobs(n_samples=1500, random_state=170, cluster_std=[1.0, 2.5, 0.5])

#
# Compute clustering with Means

n_iters = 1000
k_means = KMeans(init='random',n_clusters=3, max_iter=n_iters, n_init=1)
k_means.fit(X)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']

k_means_cluster_centers = k_means.cluster_centers_

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_8_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_8.png')
plt.show()

#
# PLOT 9 - difficult cluster type 3
#

X, labels_true = make_moons(n_samples=1500, noise=0.05)

#
# Compute clustering with Means

n_iters = 1000
k_means = KMeans(init='random',n_clusters=2, max_iter=n_iters, n_init=1)
k_means.fit(X)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34']
    #colors = ['cyan', 'magenta', 'yellow']
else:
    colors = ['#000000', '#DDDDDD']

k_means_cluster_centers = k_means.cluster_centers_

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

for k, col in zip(range(2), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('K-means %2d iteraation jälkeen' %n_iters)
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_9_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_kmeans_9.png')
plt.show()
