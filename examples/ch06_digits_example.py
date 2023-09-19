# THIS CODE IS MODIFIED VERSION OF SCIKIT-LEARN CODE
# Original Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#                   Olivier Grisel <olivier.grisel@ensta.org>
#                   Mathieu Blondel <mathieu@mblondel.org>
#                   Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2011

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

from pathlib import Path

print(__doc__)

# Book stylebook
plt.style.use('book.mplstyle')
import gaud_sty

import importlib.util
#spec = importlib.util.spec_from_file_location("sklearn_som.som", "/home/kamarain/Work/ext/sklearn-som/sklearn_som/som.py")
spec = importlib.util.spec_from_file_location("sklearn_som.som", Path.home()/"Work/ext/sklearn-som/sklearn_som/som.py")
sklearn_som = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sklearn_som)

digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30

print(X.shape)


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        if gaud_sty.color==True:
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                     color=plt.cm.Set1(y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        else:
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                     color=plt.cm.Greys(y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
            
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
            if np.min(dist) < 5e-3: # Change this to plot more digit examples 
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# ----------------------------------------------------------------------
# Plot images of the digits
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('Esimerkkejä 8x8 Digits-datajoukosta')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_digits_example_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_digits_example_1.png')
plt.show()


# ----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
#rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
rp = random_projection.GaussianRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Satunnaisprojektio")
print(X_projected)
#from pprint import pprint
#pprint(vars(X_projected))
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_digits_example_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_digits_example_2.png')
plt.show()

# ----------------------------------------------------------------------
# MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
plot_embedding(X_mds,
               "MDS-projektio")
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_digits_example_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_digits_example_3.png')
plt.show()

# ----------------------------------------------------------------------
# SOM  embedding of the digits dataset
print("Computing SOM embedding")
print(X.size)
som = sklearn_som.SOM(m=20, n=20, dim=64)
t0 = time()
som.fit(X)
bmus = som.predict(X)
X_som = som._locations[bmus,:]

#from pprint import pprint
#pprint(vars(som))


#print(X_som)

print("Done.")
plot_embedding(X_som,
               "SOM-kartta")
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_digits_example_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_digits_example_4.png')
plt.show()

# LATEEEEEEEER
## ----------------------------------------------------------------------
## t-SNE embedding of the digits dataset
#print("Computing t-SNE embedding")
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#t0 = time()
#X_tsne = tsne.fit_transform(X)
#
#plot_embedding(X_tsne,
#               "t-SNE embedding of the digits (time %.2fs)" %
#               (time() - t0))

