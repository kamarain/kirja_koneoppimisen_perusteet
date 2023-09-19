import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import gaud_sty

# Generate sample data
#np.random.seed(13)

np.random.seed(42)

mu1 = [15, 1000]
mu2 = [42, 40000]
mu3 = [90, 26000]
N1 = 300
N2 = 150
N3 = 290
cov1 = [[2**2,0],[0,250**2]]
cov2 = [[13**2,0],[0,4000**2]]
cov3 = [[8**2,0],[0,3000**2]]

X1 = np.random.multivariate_normal(mu1, cov1, N1).T
X2 = np.random.multivariate_normal(mu2, cov2, N2).T
X3 = np.random.multivariate_normal(mu3, cov3, N3).T
X = np.concatenate((X1,X2,X3),axis=1)
plt.figure(figsize=(10,4))
if gaud_sty.color==True:
    plt.plot(X[0,:],X[1,:], 'w', markerfacecolor='black', marker='.', markersize=10)
else:
    plt.plot(X[0,:],X[1,:], 'w', markerfacecolor='black', marker='.', markersize=10)
plt.xlabel('ikä [v]')
plt.ylabel('vuositulot [hopeasirppiä]')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_cluster_example_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_cluster_example_1.png')
plt.show()

plt.figure(figsize=(10,4))
if gaud_sty.color==True:
    plt.plot(X[0,:],X[1,:], 'w', markerfacecolor='black', marker='.',markersize=10)
    plt.xlabel('ikä [v]')
    plt.ylabel('vuositulot [hopeasirppiä]')
    plt.plot(mu1[0], mu1[1], 'o', markerfacecolor='red', markeredgecolor='k', markersize=14)
    plt.plot(mu2[0], mu2[1], 'o', markerfacecolor='green', markeredgecolor='k', markersize=14)
    plt.plot(mu3[0], mu3[1], 'o', markerfacecolor='orange', markeredgecolor='k', markersize=14)
    plt.axis()
    plt.text(mu1[0]+0,mu1[1]+3000,  '(%.1f, %.1f)' % (mu1[0],mu1[1]), fontsize=14)
    plt.text(mu2[0]+5,mu2[1]+9000,  '(%.1f, %.1f)' % (mu2[0],mu2[1]), fontsize=14)
    plt.text(mu3[0]+3,mu3[1]+10000,  '(%.1f, %.1f)' % (mu3[0],mu3[1]), fontsize=14)
else:
    plt.plot(X[0,:],X[1,:], 'w', markerfacecolor='black', marker='.',markersize=10)
    plt.xlabel('ikä [v]')
    plt.ylabel('vuositulot [hopeasirppiä]')
    plt.plot(mu1[0], mu1[1], 'o', markerfacecolor='black', markeredgecolor='k', markersize=14)
    plt.plot(mu2[0], mu2[1], 'o', markerfacecolor='black', markeredgecolor='k', markersize=14)
    plt.plot(mu3[0], mu3[1], 'o', markerfacecolor='black', markeredgecolor='k', markersize=14)
    plt.axis()
    plt.text(mu1[0]+0,mu1[1]+3000,  '(%.1f, %.1f)' % (mu1[0],mu1[1]), color='black',fontsize=14)
    plt.text(mu2[0]+5,mu2[1]+9000,  '(%.1f, %.1f)' % (mu2[0],mu2[1]), color='black',fontsize=14)
    plt.text(mu3[0]+3,mu3[1]+10000,  '(%.1f, %.1f)' % (mu3[0],mu3[1]), color='black',fontsize=14)
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_cluster_example_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_cluster_example_2.png')
plt.show()
