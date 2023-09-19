import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import scipy.stats as stats
from numpy.linalg import inv
plt.style.use('book.mplstyle')
import gaud_sty

#
# 1. Generate and plot random points for training
np.random.seed(66) # to always get the same points
N_h = 1000
N_e = 200
mu_h_gt = 1.1
mu_e_gt = 1.9
sigma_h_gt = 0.3
sigma_e_gt = 0.4
x_h = np.random.normal(mu_h_gt,sigma_h_gt,N_h)
x_e = np.random.normal(mu_e_gt,sigma_e_gt,N_e)
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]), linestyle='', color='cyan', label="maahinen",
             marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]), linestyle='', color='magenta', label="haltija",
             marker=gaud_sty.marker3[1])
else:
    plt.plot(x_h,np.zeros([N_h,1]), linestyle='', label="maahinen",
             c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]), linestyle='', label="haltija",
             c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
plt.title(f'{N_h} maahisen mitattu pituus')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.0,3.0,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_1.png')
plt.show()

#
#
mu_h_est = np.mean(x_h)
mu_e_est = np.mean(x_e)
sigma_h_est = np.std(x_h)
sigma_e_est = np.std(x_e)
print(f'Avg height hobits {mu_h_est:0.2f} (GT: {mu_h_gt:0.2f})')
print(f'Avg height elves {mu_e_est:0.2f} (GT: {mu_e_gt:0.2f})')
print(f'Height st. deviation hobits {sigma_h_est:0.3f} (GT: {sigma_h_gt:0.3f})')
print(f'Height st. deviation elves {sigma_e_est:0.3f} (GT: {sigma_e_gt:0.3f})')

def gaussian(x, mu, sigma, priori):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        z[i] = priori*1/np.sqrt(2*np.pi)*1/sigma*np.exp((-1/2*(x[i]-mu)**2)/(2*sigma**2))
    return z


#
#
[x, step_size] = np.linspace(0,3.0,70,retstep=True)
lhood_h_est = gaussian(x, mu_h_est, sigma_h_est, 1)
lhood_e_est = gaussian(x, mu_e_est, sigma_e_est, 1)
lhood_h_gt = gaussian(x, mu_h_gt, sigma_h_gt, 1)
lhood_e_gt = gaussian(x, mu_e_gt, sigma_e_gt, 1)
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
    plt.plot(x_e,np.zeros([N_e,1]),'mo', label="haltija")
    plt.plot(x,lhood_h_gt,'c-', label="hobitti (GT)")
    plt.plot(x,lhood_e_gt,'m-', label="haltija (GT)")
    plt.plot(x,lhood_h_est,'c--', label="hobitti (est)")
    plt.plot(x,lhood_e_est,'m--', label="haltija (est)")
else:
    plt.plot(x_h,np.zeros([N_h,1]),label="maahinen",
                          c=gaud_sty.gray2[0], linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]),label="haltija",
                          c=gaud_sty.gray2[1], linestyle='', marker=gaud_sty.marker3[1])
    plt.plot(x,lhood_h_gt, label="maahinen (GT)",
                          c=gaud_sty.gray2[0], linestyle=gaud_sty.line6[0])
    plt.plot(x,lhood_e_gt, label="haltija (GT)",
                          c=gaud_sty.gray2[1], linestyle=gaud_sty.line6[1])
    plt.plot(x,lhood_h_est, label="maahinen (est)",
                          c=gaud_sty.gray2[0], linestyle=gaud_sty.line6[2])
    plt.plot(x,lhood_e_est, label="haltija (est)",
                          c=gaud_sty.gray2[1], linestyle=gaud_sty.line6[3])

plt.legend()
plt.xlabel('pituus [m]')
#plt.axis([0.0,3.0,-1.1,+5])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_2.png')
plt.show()

#
#
kern_width = 0.2
[x, step_size] = np.linspace(0,3.0,70,retstep=True)
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
    plt.plot(x_e,np.zeros([N_e,1]),'mo', label="haltija")
else:
    plt.plot(x_h,np.zeros([N_h,1]),label="maahinen",
             c=gaud_sty.gray2[0], linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]),label="haltija",
             c=gaud_sty.gray2[1], linestyle='', marker=gaud_sty.marker3[1])

[x_kern, step_size_kern] = np.linspace(0,3.0,11,retstep=True)
x_kern_plot = np.linspace(-1.0,+4.0,201)
if gaud_sty.color==True:
    for foo_ind, foo_val in enumerate(x_kern):
        foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
        plt.plot(x_kern_plot, foo_kern,'y--',label='kerneli')
        break
else:
    for foo_ind, foo_val in enumerate(x_kern):
        foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
        plt.plot(x_kern_plot, foo_kern,label='kerneli',
                 c=gaud_sty.gray2[1], linestyle=gaud_sty.line6[1])
        break
plt.legend()
plt.xlabel('pituus [m]')
#plt.axis([0.0,3.0,-1.1,+5])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_3.png')
plt.show()

# Output value is Gaussian kernel multiplied by all positive samples
lhood_h_est_kern = np.zeros(len(x))
for xind, xval in enumerate(x):
    lhood_h_est_kern[xind] = sum(gaussian(x_h,xval,kern_width,1))/len(x_h)
    #lhood_h_est_kern[xind] = sum(stats.norm.pdf(x_h, xval, kernel_width))

if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
    plt.plot(x_e,np.zeros([N_e,1]),'mo', label="haltija")
    plt.plot(x,lhood_h_gt,'c-', label="hobitti (GT)")
    plt.plot(x,lhood_h_est_kern,'c--', label="hobitti (est)")
else:
    plt.plot(x_h,np.zeros([N_h,1]),label="maahinen",
             c=gaud_sty.gray2[0], linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]),label="haltija",
             c=gaud_sty.gray2[1], linestyle='', marker=gaud_sty.marker3[1])
    plt.plot(x,lhood_h_gt, label="maahinen (GT)",
                 c=gaud_sty.gray2[0], linestyle=gaud_sty.line5[0])
    plt.plot(x,lhood_h_est_kern, label="maahinen (est)",
                 c=gaud_sty.gray2[0], linestyle=gaud_sty.line5[2])

#plt.plot(x,lhood_e_est,'m--', label="haltija (est)")
plt.legend()
plt.xlabel('pituus [m]')
#plt.axis([0.0,3.0,-1.1,+5])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_4.png')
plt.show()

#
#
kern_width = 0.02
[x, step_size] = np.linspace(0,3.0,70,retstep=True)
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
    plt.plot(x_e,np.zeros([N_e,1]),'mo', label="haltija")
else:
    plt.plot(x_h,np.zeros([N_h,1]),label="maahinen",
             c=gaud_sty.gray2[0], linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]),label="haltija",
             c=gaud_sty.gray2[1], linestyle='', marker=gaud_sty.marker3[1])
[x_kern, step_size_kern] = np.linspace(0,3.0,11,retstep=True)
x_kern_plot = np.linspace(-1.0,+4.0,201)
if gaud_sty.color==True:
    for foo_ind, foo_val in enumerate(x_kern):
        foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
        plt.plot(x_kern_plot, foo_kern,'y--',label='kerneli')
        break
else:
    for foo_ind, foo_val in enumerate(x_kern):
        foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
        plt.plot(x_kern_plot, foo_kern,label='kerneli',
                 c=gaud_sty.gray2[1], linestyle=gaud_sty.line6[1])
        break
plt.legend()
plt.xlabel('pituus [m]')
#plt.axis([0.0,3.0,-1.1,+5])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_5_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_5.png')
plt.show()

# Output value is Gaussian kernel multiplied by all positive samples
lhood_h_est_kern = np.zeros(len(x))
for xind, xval in enumerate(x):
    lhood_h_est_kern[xind] = sum(gaussian(x_h,xval,kern_width,1))/len(x_h)
    #lhood_h_est_kern[xind] = sum(stats.norm.pdf(x_h, xval, kernel_width))

if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
    plt.plot(x_e,np.zeros([N_e,1]),'mo', label="haltija")
    plt.plot(x,lhood_h_gt,'c-', label="hobitti (GT)")
    plt.plot(x,lhood_h_est_kern,'c--', label="hobitti (est)")
else:
    plt.plot(x_h,np.zeros([N_h,1]),label="maahinen",
             c=gaud_sty.gray2[0], linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]),label="haltija",
             c=gaud_sty.gray2[1], linestyle='', marker=gaud_sty.marker3[1])
    plt.plot(x,lhood_h_gt, label="maahinen (GT)",
                 c=gaud_sty.gray2[0], linestyle=gaud_sty.line5[0])
    plt.plot(x,lhood_h_est_kern, label="maahinen (est)",
                 c=gaud_sty.gray2[0], linestyle=gaud_sty.line5[2])

#plt.plot(x,lhood_e_est,'m--', label="haltija (est)")
plt.legend()
plt.xlabel('pituus [m]')
#plt.axis([0.0,3.0,-1.1,+5])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_6_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_6.png')
plt.show()

#
#
kern_width = 0.8
[x, step_size] = np.linspace(0,3.0,70,retstep=True)
[x_kern, step_size_kern] = np.linspace(0,3.0,11,retstep=True)
x_kern_plot = np.linspace(-1.0,+4.0,201)
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
    plt.plot(x_e,np.zeros([N_e,1]),'mo', label="haltija")
    for foo_ind, foo_val in enumerate(x_kern):
        foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
        plt.plot(x_kern_plot, foo_kern,'y--',label='kerneli')
        break
else:
    plt.plot(x_h,np.zeros([N_h,1]),label="maahinen",
             c=gaud_sty.gray2[0], linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]),label="haltija",
             c=gaud_sty.gray2[1], linestyle='', marker=gaud_sty.marker3[1])
    for foo_ind, foo_val in enumerate(x_kern):
        foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
        plt.plot(x_kern_plot, foo_kern,label='kerneli',
                 c=gaud_sty.gray2[1], linestyle=gaud_sty.line6[1])
        break
#for foo_ind, foo_val in enumerate(x_kern):
#    foo_kern = gaussian(x_kern_plot, foo_val, kern_width, 1)
#    plt.plot(x_kern_plot, foo_kern,'y--',label='kerneli')
#    break
plt.legend()
plt.xlabel('pituus [m]')
#plt.axis([0.0,3.0,-1.1,+5])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_7_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_7.png')
plt.show()

# Output value is Gaussian kernel multiplied by all positive samples
lhood_h_est_kern = np.zeros(len(x))
for xind, xval in enumerate(x):
    lhood_h_est_kern[xind] = sum(gaussian(x_h,xval,kern_width,1))/len(x_h)
    #lhood_h_est_kern[xind] = sum(stats.norm.pdf(x_h, xval, kernel_width))

if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
    plt.plot(x_e,np.zeros([N_e,1]),'mo', label="haltija")
    plt.plot(x,lhood_h_gt,'c-', label="hobitti (GT)")
    plt.plot(x,lhood_h_est_kern,'c--', label="hobitti (est)")
else:
    plt.plot(x_h,np.zeros([N_h,1]),label="maahinen",
             c=gaud_sty.gray2[0], linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N_e,1]),label="haltija",
             c=gaud_sty.gray2[1], linestyle='', marker=gaud_sty.marker3[1])
    plt.plot(x,lhood_h_gt, label="maahinen (GT)",
                 c=gaud_sty.gray2[0], linestyle=gaud_sty.line5[0])
    plt.plot(x,lhood_h_est_kern, label="maahinen (est)",
                 c=gaud_sty.gray2[0], linestyle=gaud_sty.line5[2])
#plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
#plt.plot(x_e,np.zeros([N_e,1]),'mo', label="haltija")
#plt.plot(x,lhood_h_gt,'c-', label="hobitti (GT)")
#plt.plot(x,lhood_h_est_kern,'c--', label="hobitti (est)")
##plt.plot(x,lhood_e_est,'m--', label="haltija (est)")
plt.legend()
plt.xlabel('pituus [m]')
#plt.axis([0.0,3.0,-1.1,+5])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_8_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_nonparametric_8.png')
plt.show()
