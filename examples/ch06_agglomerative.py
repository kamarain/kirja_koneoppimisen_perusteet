print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

import gaud_sty
plt.style.use('book.mplstyle')


#
# PLOTS 1 - single linkage
#


# A suitable seed for nice pictures
#np.random.seed(666) # very bad
#np.random.seed(3) # very good
#np.random.seed(4) # super good
agglo_linkage = 'single' # single / ward / maximum / average


# Generate sample data: easy

centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

#
# Compute clustering with hierarchial agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=3, linkage=agglo_linkage, connectivity=None)

agglo.fit(X)
y_pred = agglo.labels_.astype(int)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #colors = ['cyan', 'magenta', 'yellow']
    markers = ['.','.','.']
else:
    colors = ['#000000', '#888888', '#DDDDDD']
    markers = ['.','.','.']


for k, col in zip(range(n_clusters), colors):
    my_members = y_pred == k
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #        markeredgecolor='k', markersize=6)
plt.title(f'Kokoava hierarkkinen ryv. (linkage={agglo_linkage})')
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_1.png')
plt.show()


#
# Generate sample data: difficult

X, labels_true = make_blobs(n_samples=500, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
X = X_aniso


#
# Compute clustering with hierarchial agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=3, linkage=agglo_linkage, connectivity=None)

agglo.fit(X)
y_pred = agglo.labels_.astype(int)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']


for k, col in zip(range(n_clusters), colors):
    my_members = y_pred == k
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #        markeredgecolor='k', markersize=6)
#plt.title(f'Kokoava hierarkkinen ryv. (linkage={agglo_linkage})')
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_2.png')
plt.show()



#
# Generate sample data: difficult 2

X, labels_true = make_blobs(n_samples=1500, random_state=170, cluster_std=[1.0, 2.5, 0.5])

#
# Compute clustering with hierarchial agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=3, linkage=agglo_linkage, connectivity=None)

agglo.fit(X)
y_pred = agglo.labels_.astype(int)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']


for k, col in zip(range(n_clusters), colors):
    my_members = y_pred == k
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #        markeredgecolor='k', markersize=6)
#plt.title(f'Kokoava hierarkkinen ryvästys (linkage={agglo_linkage})')
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_3.png')
plt.show()


#
# Generate sample data: difficult 3

X, labels_true = make_moons(n_samples=1500, noise=0.05)

#
# Compute clustering with hierarchial agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=2, linkage=agglo_linkage, connectivity=None)

agglo.fit(X)
y_pred = agglo.labels_.astype(int)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']


for k, col in zip(range(n_clusters), colors):
    my_members = y_pred == k
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #        markeredgecolor='k', markersize=6)
#plt.title(f'Kokoava hierarkkinen ryvästys (linkage={agglo_linkage})')
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_4.png')
plt.show()



#
# PLOTS 1 - single linkage
#


# A suitable seed for nice pictures
#np.random.seed(666) # very bad
#np.random.seed(3) # very good
#np.random.seed(4) # super good
agglo_linkage = 'ward' # single / ward / maximum / average


# Generate sample data: easy

centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

#
# Compute clustering with hierarchial agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=3, linkage=agglo_linkage, connectivity=None)

agglo.fit(X)
y_pred = agglo.labels_.astype(int)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']
if gaud_sty.color==True:
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #colors = ['cyan', 'magenta', 'yellow']
    markers = ['.','.','.']
else:
    colors = ['#000000', '#888888', '#DDDDDD']
    markers = ['.','.','.']


for k, col in zip(range(n_clusters), colors):
    my_members = y_pred == k
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #        markeredgecolor='k', markersize=6)
plt.title(f'Kokoava hierarkkinen ryv. (linkage={agglo_linkage})')
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_5_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_5.png')
plt.show()


#
# Generate sample data: difficult

X, labels_true = make_blobs(n_samples=500, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
X = X_aniso


#
# Compute clustering with hierarchial agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=3, linkage=agglo_linkage, connectivity=None)

agglo.fit(X)
y_pred = agglo.labels_.astype(int)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']


for k, col in zip(range(n_clusters), colors):
    my_members = y_pred == k
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #        markeredgecolor='k', markersize=6)
#plt.title(f'Kokoava hierarkkinen ryvästys (linkage={agglo_linkage})')
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_6_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_6.png')
plt.show()



#
# Generate sample data: difficult 2

X, labels_true = make_blobs(n_samples=1500, random_state=170, cluster_std=[1.0, 2.5, 0.5])

#
# Compute clustering with hierarchial agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=3, linkage=agglo_linkage, connectivity=None)

agglo.fit(X)
y_pred = agglo.labels_.astype(int)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']


for k, col in zip(range(n_clusters), colors):
    my_members = y_pred == k
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #        markeredgecolor='k', markersize=6)
#plt.title(f'Kokoava hierarkkinen ryvästys (linkage={agglo_linkage})')
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_7_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_7.png')
plt.show()


#
# Generate sample data: difficult 3

X, labels_true = make_moons(n_samples=1500, noise=0.05)

#
# Compute clustering with hierarchial agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=2, linkage=agglo_linkage, connectivity=None)

agglo.fit(X)
y_pred = agglo.labels_.astype(int)

#
# Plot result

fig = plt.figure(figsize=(4, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#colors = ['cyan', 'magenta', 'yellow']


for k, col in zip(range(n_clusters), colors):
    my_members = y_pred == k
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #        markeredgecolor='k', markersize=6)
#plt.title(f'Kokoava hierarkkinen ryvästys (linkage={agglo_linkage})')
plt.axis('equal')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_8_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_agglomerative_8.png')
plt.show()
