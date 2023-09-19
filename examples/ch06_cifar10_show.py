import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import gaud_sty
from pathlib import Path

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# relative to your home directory
#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/data_batch_1')
datadict = unpickle(Path.home()/'Data/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle(Path.home()/'Data/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

#plt.figure(figsize=(10,10))
fig, axs = plt.subplots(nrows=10,ncols=10, figsize=(6, 6))
plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.999, wspace=.02, hspace=.02)
#fig,axs = plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
#                    hspace=.01)
plot_row_num = 0
plot_col_num = 0

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.99:
        #plt.figure(1);
        #plt.clf()
        axs[plot_row_num,plot_col_num].axis('off')
        if gaud_sty.color==True:
            axs[plot_row_num,plot_col_num].imshow(X[i])
        else:
            foo = X[i]
            if gaud_sty.color==True:
                axs[plot_row_num,plot_col_num].imshow(np.dot(foo[...,:3], [0.2989, 0.5870, 0.1140]))
            else:
                axs[plot_row_num,plot_col_num].imshow(np.dot(foo[...,:3], [0.2989, 0.5870, 0.1140]),cmap='gray')

        #plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        plot_col_num = plot_col_num+1
        if plot_col_num == 10:
            plot_row_num = plot_row_num+1
            plot_col_num = 0
        if plot_row_num == 10:
            break
        
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch06_cifar10_show_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch06_cifar10_show_1.png')
plt.show()
plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
