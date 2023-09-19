import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.linalg import inv
plt.style.use('book.mplstyle')
import gaud_sty

#
# 1. Generate and plot random points for training
np.random.seed(13) # to always get the same points
N_h = 1000
N_e = 200
x_h = np.random.normal(1.1,0.3,N_h)
x_e = np.random.normal(1.9,0.4,N_e)
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N_h,1]), 'ko', label="maahinen")
else:
    plt.plot(x_h,np.zeros([N_h,1]),'ko', label="maahinen")
#plt.plot(x_e,np.zeros([N,1]),'mo', label="haltija")
plt.title(f'{N_h} maahisen mitattu pituus')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.0,3.0,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_1.png')
plt.show()

#
# 2. Histogram plots with 5 bins
if gaud_sty.color==True:
    foo = plt.hist(x_h, bins = 5, range = [0,2.5])
else:
    foo = plt.hist(x_h, bins = 5, range = [0,2.5], color='gray')
x_h_hist = foo[0]
x_h_hist_bins = foo[1]
print(x_h_hist)
print(x_h_hist_bins)
plt.xlabel('pituus [m]')
plt.ylabel('lukumäärä [kpl]')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_2.png')
plt.show()

if gaud_sty.color==True:
    foo = plt.hist(x_h_hist_bins[:-1], x_h_hist_bins, weights=x_h_hist/1000)
else:
    foo = plt.hist(x_h_hist_bins[:-1], x_h_hist_bins, weights=x_h_hist/1000, color='gray')
x_h_hist = foo[0]
x_h_hist_bins = foo[1]
print(x_h_hist)
print(x_h_hist_bins)
plt.xlabel('pituus [m]')
plt.ylabel('todennäköisyys')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_3.png')
plt.show()

#foo = plt.hist(x_h_hist_bins[:-1], x_h_hist_bins, weights=x_h_hist/1000)
if gaud_sty.color==True:
    foo = plt.hist(x_h, bins = 5, range = [0,2.5], density=True, rwidth=0.1)
else:
    foo = plt.hist(x_h, bins = 5, range = [0,2.5], density=True, rwidth=0.1, color='gray')
x_h_hist = foo[0]
x_h_hist_bins = foo[1]
print(x_h_hist)
print(x_h_hist_bins)
plt.xlabel('pituus [m]')
plt.ylabel('todennäköisyystiheys')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_4.png')
plt.show()



#
# 3. Plots with more bins
foo = plt.hist(x_h, bins = 11, range = [0,2.5])
x_h_hist = foo[0]
x_h_hist_bins = foo[1]
print(x_h_hist)
print(x_h_hist_bins)
plt.xlabel('pituus [m]')
plt.ylabel('lukumäärä [kpl]')
plt.show()
foo = plt.hist(x_h, bins = 21, range = [0,2.5])
x_h_hist = foo[0]
x_h_hist_bins = foo[1]
print(x_h_hist)
print(x_h_hist_bins)
plt.xlabel('pituus [m]')
plt.ylabel('lukumäärä [kpl]')
plt.show()
foo = plt.hist(x_h, bins = 21, density=True, range = [0,2.5])
x_h_hist = foo[0]
x_h_hist_bins = foo[1]
print(x_h_hist)
print(x_h_hist_bins)
plt.xlabel('pituus [m]')
plt.ylabel('tiheys')
plt.show()
#x_h_hist, x_h_edges = np.histogram(x_h,bins=11)
#plt.plot(x_h,np.zeros([N_h,1]),'co', label="hobitti")
#plt.plot(x_e,np.zeros([N,1]),'mo', label="haltija")
#plt.title(f'{N_h} hobitin mitattu pituus')
#plt.legend()
plt.xlabel('pituus [m]')
plt.ylabel('lukumäärä [kpl]')
#plt.axis([0.0,3.0,-1.1,+1.1])
plt.savefig(gaud_sty.save_dir+'ch04_cond_probability_5.png')
plt.show()


#
# 2. Add y values that represent the two classes and plot
y_h = np.zeros(N)
y_h[:] = -1.0
y_e = np.zeros(N)
y_e[:] = +1.0
plt.plot(x_h,y_h,'co', label="hobitti")
plt.plot(x_e,y_e,'mo', label="haltija")
plt.title('Näytteitä kahdesta luokasta (c1 = -1, c2 = +1)')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,2.5,-1.1,+1.1])
plt.show()

#
# 3. Fit and plot line

# Form the train input and output vectors  (1: hobit, 2: elf)
x_tr = np.concatenate((x_h,x_e))
y_tr = np.concatenate((y_h,y_e))

# Matrix operations
X = np.concatenate((np.transpose([np.ones(2*N)]),np.transpose([x_tr])), axis=1)
w_foo = np.matmul(np.transpose(X),X)
w_foo = inv(w_foo)
w = np.matmul(np.matmul(y_tr,X),w_foo)

a = w[1]
b = w[0]
y_hat_tr = a*x_tr+b
MSE = np.sum((y_tr-y_hat_tr)**2)/N

# Coodinate system
plt.plot(x_h,y_h,'co', label="hobitti")
plt.plot(x_e,y_e,'mo', label="haltija")
plt.legend()
plt.plot(x_tr,a*x_tr+b,'b-')
plt.title(f"Sovitettu suora (a={a:.1f}, b={b:.1f}, MSE={MSE:.1f})")
plt.xlabel('pituus [m]')
plt.axis([0.5,2.5,-1.1,+1.1])
plt.show()

#
# 4. Generate and plot random points for testing
N_t =  3
x_h_test = np.random.normal(1.1,0.3,N_t) # h as hobit
x_e_test = np.random.normal(1.9,0.4,N_t) # e as elf
plt.plot(x_h,y_h,'co', label="hobitti")
plt.plot(x_e,y_e,'mo', label="haltija")
plt.legend()
plt.plot(x_tr,a*x_tr+b,'b-')
plt.plot(x_e_test,np.zeros([N_t,1]),'kv', label="tuntematon")
plt.plot(x_h_test,np.zeros([N_t,1]),'kv')
plt.title('Luokiteltavia näytteitä')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,2.5,-1.1,+1.1])
plt.show()

#
# 5. Do classification and plot results
x_te = np.concatenate((x_h_test,x_e_test))
y_te = np.concatenate((1*np.ones([N_t,1]),2*np.ones([N_t,1])))
y_te_pred = a*x_te+b
plt.plot(x_h,y_h,'co', label="maahinen")
plt.plot(x_e,y_e,'mo', label="haltija")
plt.legend()
plt.plot(x_tr,a*x_tr+b,'b-')

corr_class = 2*N_t
for s_ind, s in enumerate(x_te):
    print(y_te[s_ind])
    if y_te_pred[s_ind] > 0:
        plt.plot(s,1,'mv')
        if y_te[s_ind] != 2: # wrong class
            plt.plot(s,1, 'rx', markersize=30)
            corr_class = corr_class-1
    else:
        plt.plot(s,-1, 'cv')
        if y_te[s_ind] != 1: # wrong class
            plt.plot(s,-1, 'rx', markersize=30)
            corr_class = corr_class-1
plt.title('Luokittelutulos (suoran sovitus)')
#plt.axis([0.5,2.5,-1.1,+1.1])
tot_correct = corr_class/(2*N_t)
print(f'Classication accuracy: {tot_correct*100:.2f}%')
plt.xlabel('pituus [m]')
plt.axis([0.5,2.5,-1.1,+1.1])
plt.show()
