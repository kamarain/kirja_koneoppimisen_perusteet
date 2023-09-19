import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import datasets

from pathlib import Path

# Book stylebook
plt.style.use('book.mplstyle')
import gaud_sty

# Set for suitable plots
rnd_seed = 666

# Import custom SOM implementation (location relative to home dir)
import importlib.util
#spec = importlib.util.spec_from_file_location("sklearn_som.som", "/home/kamarain/Work/ext/sklearn-som/sklearn_som/som.py")
spec = importlib.util.spec_from_file_location("sklearn_som.som", Path.home()/"Work/ext/sklearn-som/sklearn_som/som.py")
sklearn_som = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sklearn_som)


n_points = 1000
X, color = datasets.make_s_curve(n_points, random_state=0, noise=0.1)
n_neighbors = 10
n_components = 2
X2d = np.concatenate(([X[:,0]], [X[:,2]]),axis=0).T

# Create figure
fig = plt.figure()
if gaud_sty.color==True:
    plt.scatter(X2d[:,0],X2d[:,1], c=color, cmap=plt.cm.Spectral)
else:
    plt.scatter(X2d[:,0],X2d[:,1], c=color, cmap=plt.cm.Greys)
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_som_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_som_1.png')
plt.show()

som = sklearn_som.SOM(m=1, n=13, dim=2, random_state=rnd_seed)
som.fit(X2d, shuffle=False)
#bmus = som.predict(X2d)
#X_som = som._locations[bmus,:]
X_w = som.weights

if gaud_sty.color==True:
    plt.scatter(X2d[:,0],X2d[:,1], c=color, cmap=plt.cm.Spectral)
else:
    plt.scatter(X2d[:,0],X2d[:,1], c=color, cmap=plt.cm.Greys)
plt.plot(X_w[:,0],X_w[:,1],'k-')
plt.plot(X_w[:,0],X_w[:,1],'ko')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_som_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_som_2.png')
plt.show()
