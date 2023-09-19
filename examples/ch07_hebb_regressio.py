import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import gaud_sty
plt.style.use('book.mplstyle')

# Coodinate system
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])

# Generate random points
np.random.seed(42) # to always get the same points
N = 50 
x = np.random.normal(1.1,0.3,N)
a_gt = 50.0
b_gt = 20.0
y_noise =  np.random.normal(0,8,N) # Measurements from the class 1\n",
y = a_gt*x+b_gt+y_noise
if gaud_sty.color==True:
    plt.plot(x,y,'ro')
else:
    plt.plot(x,y,'ko')
    
plt.title('Opetusaineisto suoran sovitukseen')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_1.png')
plt.show()

# Compute MSE heat map for different a and b
a_t = 0
b_t = 0
num_of_epochs = 10
learning_rate = 0.005

y_h = a_t*x+b_t
MSE = np.sum((y-y_h)**2)/N
plt.title(f'Epoch={0} a={a_t:.2f} b={b_t:.2f} MSE={MSE:.2f}')
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])
if gaud_sty.color==True:
    plt.plot(x,y,'ro')
    plt.plot(x,a_t*x+b_t,'k-')
else:
    plt.plot(x,y,'ko')
    plt.plot(x,a_t*x+b_t,'k-')

if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_2.png')
plt.show()

plotind = 0
for e in range(num_of_epochs):
    for x_e_ind,x_e in enumerate(x):
        y_e = a_t*x_e+b_t
        a_t = a_t+learning_rate*(y[x_e_ind]-y_e)*x_e
        b_t = b_t+learning_rate*(y[x_e_ind]-y_e)*x_e
    # Compute train error
    y_h = a_t*x+b_t
    MSE = np.sum((y-y_h)**2)/N
    plt.title(f'Epoch={e+1} a={a_t:.2f} b={b_t:.2f} MSE={MSE:.2f}')
    plt.xlabel('pituus [m]')
    plt.ylabel('paino [kg]')
    plt.axis([0,2,0,150])
    if e in [0,4,9]:
        if gaud_sty.color==True:
            plt.plot(x,y,'ro')
            plt.plot(x,a_t*x+b_t,'k-')
        else:
            plt.plot(x,y,'ko')
            plt.plot(x,a_t*x+b_t,'k-')
        if plotind == 0:
            if gaud_sty.savefig:
                if gaud_sty.color:
                    plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_3_col.png')
                else:
                    plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_3.png')
            plotind += 1
        elif plotind == 1:
            if gaud_sty.color:
                plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_4_col.png')
            else:
                plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_4.png')
            plotind += 1
        elif plotind == 2:
            if gaud_sty.color:
                plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_5_col.png')
            else:
                plt.savefig(gaud_sty.save_dir+'ch07_hebb_regressio_5.png')
            plotind += 1
        plt.show()
