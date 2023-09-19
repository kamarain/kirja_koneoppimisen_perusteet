import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.linalg import inv
plt.style.use('book.mplstyle')
import gaud_sty

#
# 1. Bernoulli parameter estimation (ML)
np.random.seed(666) # to always get the same points

x = np.linspace(-6.0,6.0,101)
mu_1 = 0.0
sigma2_1 = 2
gauss_1 = 1/np.sqrt(2*np.pi*sigma2_1)*np.exp(-1/(2*sigma2_1)*(x-mu_1)**2)
mu_2 = 2.0
sigma2_2 = 2
gauss_2 = 1/np.sqrt(2*np.pi*sigma2_2)*np.exp(-1/(2*sigma2_2)*(x-mu_2)**2)
mu_3 = 2.0
sigma2_3 = 4
gauss_3 = 1/np.sqrt(2*np.pi*sigma2_3)*np.exp(-1/(2*sigma2_3)*(x-mu_3)**2)


#
# Plot 1
if gaud_sty.color==True:
    plt.plot(x, gauss_1, label='mu=0.0, sigma2=2.0',linestyle=gaud_sty.line5[0])
    plt.plot(x, gauss_2, label='mu=2.0, sigma2=2.0', linestyle=gaud_sty.line5[1])
    plt.plot(x, gauss_3, label='mu=2.0, sigma2=4.0', linestyle=gaud_sty.line5[2])
else:
    plt.plot(x, gauss_1, label='mu=0.0, sigma2=2.0',linestyle=gaud_sty.line5[0], c=gaud_sty.gray5[0])
    plt.plot(x, gauss_2, label='mu=2.0, sigma2=2.0', linestyle=gaud_sty.line5[1], c=gaud_sty.gray5[1])
    plt.plot(x, gauss_3, label='mu=2.0, sigma2=4.0', linestyle=gaud_sty.line5[2], c=gaud_sty.gray5[2])
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian_1.png')
plt.show()
