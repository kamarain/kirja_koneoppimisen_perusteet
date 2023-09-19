import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.linalg import inv
import gaud_sty

# Book stylebook
plt.style.use('book.mplstyle')

#
# 1. Generate and plot random points for training
np.random.seed(13) # to always get the same points
N = 5
x_h = np.random.normal(1.1,0.3,N)
x_e = np.random.normal(1.9,0.4,N)
x_e = np.append(x_e, [4.9]) # add giant elf
#plt.plot(x_h,np.zeros([N,1]),'co', label="hobitti")
#plt.plot(x_e,np.zeros([N+1,1]),'mo', label="haltija")
if gaud_sty.color==True:
    plt.plot(x_h,np.zeros([N,1]), linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N+1,1]), linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
else:
    plt.plot(x_h,np.zeros([N,1]), linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,np.zeros([N+1,1]), linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
plt.title('Opetusnäytteitä kahdesta luokasta c1 ja c2')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_1.png')
plt.show()


#
# 2. Add y values that represent the two classes and plot
y_h = np.zeros(N)
y_h[:] = -1.0
y_e = np.zeros(N+1)
y_e[:] = +1.0
#plt.plot(x_h,y_h,'co', label="hobitti")
#plt.plot(x_e,y_e,'mo', label="haltija")
if gaud_sty.color==True:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
else:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
plt.title('Näytteitä kahdesta luokasta (c1 = -1, c2 = +1)')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_2.png')
plt.show()

#
# 3. Fit and plot line

# Form the train input and output vectors  (1: hobit, 2: elf)
x_tr = np.concatenate((x_h,x_e))
y_tr = np.concatenate((y_h,y_e))

# Matrix operations
X = np.concatenate((np.transpose([np.ones(2*N+1)]),np.transpose([x_tr])), axis=1)
w_foo = np.matmul(np.transpose(X),X)
w_foo = inv(w_foo)
w = np.matmul(np.matmul(y_tr,X),w_foo)

a = w[1]
b = w[0]
y_hat_tr = a*x_tr+b
MSE = np.sum((y_tr-y_hat_tr)**2)/(N+1)

# Coodinate system
#plt.plot(x_h,y_h,'co', label="hobitti")
#plt.plot(x_e,y_e,'mo', label="haltija")
if gaud_sty.color==True:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
    plt.legend()
    plt.plot(x_tr,a*x_tr+b,'b-')
else:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    plt.legend()
    plt.plot(x_tr,a*x_tr+b,'k-')
plt.title(f"Sovitettu suora (a={a:.1f}, b={b:.1f}, MSE={MSE:.1f})")
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_3.png')
plt.show()

#
# 4. Generate and plot random points for testing
N_t =  3
x_h_test = np.random.normal(1.1,0.3,N_t) # h as hobit
x_e_test = np.random.normal(1.9,0.4,N_t) # e as elf
#plt.plot(x_h,y_h,'co', label="hobitti")
#plt.plot(x_e,y_e,'mo', label="haltija")
if gaud_sty.color==True:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
    plt.legend()
    plt.plot(x_tr,a*x_tr+b,'b-')
    #plt.plot(x_e_test,np.zeros([N_t,1]),'kv', label="tuntematon")
    #plt.plot(x_h_test,np.zeros([N_t,1]),'kv')
    plt.plot(x_e_test,np.zeros([N_t,1]), label="tuntematon", linestyle='', c='orange', marker=gaud_sty.marker3[2])
    plt.plot(x_h_test,np.zeros([N_t,1]), linestyle='', c='orange', marker=gaud_sty.marker3[2])
else:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    plt.legend()
    plt.plot(x_tr,a*x_tr+b,'k-')
    #plt.plot(x_e_test,np.zeros([N_t,1]),'kv', label="tuntematon")
    #plt.plot(x_h_test,np.zeros([N_t,1]),'kv')
    plt.plot(x_e_test,np.zeros([N_t,1]), label="tuntematon", linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[2])
    plt.plot(x_h_test,np.zeros([N_t,1]), linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[2])
plt.title('Luokiteltavia näytteitä')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_4.png')
plt.show()

#
# 5. Do classification and plot results
x_te = np.concatenate((x_h_test,x_e_test))
y_te = np.concatenate((1*np.ones([N_t,1]),2*np.ones([N_t,1])))
y_te_pred = a*x_te+b
#plt.plot(x_h,y_h,'co', label="hobitti")
#plt.plot(x_e,y_e,'mo', label="haltija")
if gaud_sty.color==True:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
    plt.legend()
    plt.plot(x_tr,a*x_tr+b,'b-')
else:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    plt.legend()
    plt.plot(x_tr,a*x_tr+b,'k-')

corr_class = 2*N_t
for s_ind, s in enumerate(x_te):
    print(y_te[s_ind])
    if y_te_pred[s_ind] > 0:
        #plt.plot(s,1,'mv')
        if gaud_sty.color==True:
            plt.plot(s,1,  linestyle='', c='orange', marker=gaud_sty.marker3[1])
        else:
            plt.plot(s,1,  linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[1])
        if y_te[s_ind] != 2: # wrong class
            plt.plot(s,1, 'kx', markersize=30)
            corr_class = corr_class-1
    else:
        #plt.plot(s,-1, 'cv')
        if gaud_sty.color==True:
            plt.plot(s,-1,  linestyle='', c='orange', marker=gaud_sty.marker3[0])
        else:
            plt.plot(s,-1,  linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[0])
        if y_te[s_ind] != 1: # wrong class
            plt.plot(s,-1, 'kx', markersize=30)
            corr_class = corr_class-1
plt.title('Luokittelutulos (suoran sovitus)')
#plt.axis([0.5,2.5,-1.1,+1.1])
tot_correct = corr_class/(2*N_t)
print(f'Classication accuracy: {tot_correct*100:.2f}%')
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_5_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_failure_5.png')
plt.show()
