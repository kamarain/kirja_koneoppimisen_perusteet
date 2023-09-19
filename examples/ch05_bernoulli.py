import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.linalg import inv
plt.style.use('book.mplstyle')
import gaud_sty

#
# 1. Bernoulli parameter estimation (ML)
np.random.seed(666) # to always get the same points

mu = 0.3
N_seq = np.array([1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50])
mu_ML = np.empty(N_seq.shape)
a=4
b=6
a2=40
b2=60
mu_MAP = np.empty(N_seq.shape)
mu_MAP2 = np.empty(N_seq.shape)
for N_num, N in enumerate(N_seq):
    x_n = np.random.binomial(1,mu,size=N)
    mu_ML[N_num] = sum(x_n)/N
    mu_MAP[N_num] = (sum(x_n)+a)/(N+a+b)
    mu_MAP2[N_num] = (sum(x_n)+a2)/(N+a2+b2)

#
# Plot 1 - only ML
fig = plt.figure(figsize =(8, 4))
if gaud_sty.color==True:
    plt.plot(N_seq, mu_ML,'--', label='mu ML')
    plt.plot([0,N],[mu,mu],'-')
else:
    plt.plot(N_seq, mu_ML,'--', c='0.5', label='mu ML')
    plt.plot([0,N],[mu,mu],'k-')
plt.xticks(N_seq)
plt.xlabel('N')
plt.ylabel('mu')
plt.legend()
#plt.gcf().subplots_adjust(top=0.8)
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_bernoulli_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_bernoulli_1.png')
plt.show()

#
# Plot 2 - ML  plus 2x MAP
fig = plt.figure(figsize =(8, 4))
if gaud_sty.color==True:
    plt.plot([0,N],[mu,mu],label='mu',linestyle=gaud_sty.line5[0])
    plt.plot(N_seq, mu_ML,label='mu ML',linestyle=gaud_sty.line5[1])
    plt.plot(N_seq, mu_MAP,label='mu MAP (a=4,b=6)',linestyle=gaud_sty.line5[2])
    plt.plot(N_seq, mu_MAP2,label='mu MAP (a=40,b=60)',linestyle=gaud_sty.line5[3])
else:
    plt.plot([0,N],[mu,mu],label='mu',linestyle=gaud_sty.line5[0], c=gaud_sty.gray5[0])
    plt.plot(N_seq, mu_ML,label='mu ML',linestyle=gaud_sty.line5[1], c=gaud_sty.gray5[1])
    plt.plot(N_seq, mu_MAP,label='mu MAP (a=4,b=6)',linestyle=gaud_sty.line5[2], c=gaud_sty.gray5[2])
    plt.plot(N_seq, mu_MAP2,label='mu MAP (a=40,b=60)',linestyle=gaud_sty.line5[3], c=gaud_sty.gray5[3])
plt.xticks(N_seq)
plt.xlabel('N')
plt.ylabel('mu')
plt.legend()
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_bernoulli_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_bernoulli_2.png')
plt.show()
