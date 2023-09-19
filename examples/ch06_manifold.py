# CODE MODIFIED FROM:
# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>

print(__doc__)

from collections import OrderedDict
from functools import partial
from time import time

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets, random_projection

# Book stylebook
plt.style.use('book.mplstyle')
import gaud_sty

# Set seed for suistable plots
np.random.seed(666)

n_points = 500
X, color = datasets.make_s_curve(n_points, random_state=0, noise=0.1)
n_neighbors = 10
n_components = 2

# Create figure
fig = plt.figure() #fig = plt.figure(figsize=(15, 8))
#fig.suptitle("Manifold Learning with %i points, %i neighbors"
#             % (1000, n_neighbors), fontsize=14)

# Add 3d scatter plot
#ax = fig.add_subplot(projection='3d')
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
#ax.view_init(4, -72)
if gaud_sty.color==True:
    plt.scatter(X[:,0],X[:,2], c=color, cmap=plt.cm.Spectral)
else:
    plt.scatter(X[:,0],X[:,2], c=color, cmap=plt.cm.Greys)
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_1.png')
plt.show()
X2d = np.concatenate(([X[:,0]], [X[:,2]]),axis=0).T
print(X2d.shape)

mds  = manifold.MDS(1, max_iter=1000, n_init=10, random_state=666)

# Plot results
t0 = time()
Y = mds.fit_transform(X2d)
t1 = time()
print("%s: %.2g sec" % ("MDS", t1 - t0))
#ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
if gaud_sty.color==True:
    plt.scatter(Y, np.zeros(Y.size), c=color, cmap=plt.cm.Spectral)
else:
    plt.scatter(Y, np.zeros(Y.size), c=color, cmap=plt.cm.Greys)
#ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_2.png')
plt.show()

## Next line to silence pyflakes. This import is needed.
Axes3D

fig = plt.figure() #fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(projection='3d')
if gaud_sty.color==True:
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
else:
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Greys)
ax.view_init(4, -50)
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_3.png')
plt.show()


mds  = manifold.MDS(2, max_iter=1000, n_init=10)

# Plot results
t0 = time()
Y = mds.fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ("MDS", t1 - t0))
#ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
if gaud_sty.color==True:
    plt.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral)
else:
    plt.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Greys)
#ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_4.png')
plt.show()

t0 = time()
rp = random_projection.GaussianRandomProjection(n_components=2)
Y = rp.fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ("Random", t1 - t0))
if gaud_sty.color==True:
    plt.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral)
else:
    plt.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Greys)
#ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_5_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_5.png')
plt.show()

# Reset for another random projection
np.random.seed(666)

t0 = time()
rp = random_projection.GaussianRandomProjection(n_components=2)
Y = rp.fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ("Random", t1 - t0))
if gaud_sty.color==True:
    plt.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral)
else:
    plt.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Greys)
#ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_6_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_manifold_6.png')
plt.show()






