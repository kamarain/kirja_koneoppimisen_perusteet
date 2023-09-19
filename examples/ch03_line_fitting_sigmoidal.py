import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.special import expit

# Book stylebook
import gaud_sty
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
plt.show()


#
# 2. Add y values that represent the two classes and plot
y_h = np.zeros(N)
y_h[:] = 0.0
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
plt.axis([0.5,5.0,-0.1,+1.1])
plt.show()


#
# 3. Fit and plot line after K epochs

# Form the train input and output vectors  (1: hobit, 2: elf)
x_tr = np.concatenate((x_h,x_e))
y_tr = np.concatenate((y_h,y_e))


# Initialize gradient descent
a_t = 0
b_t = 0
num_of_epochs = 100
learning_rate = 0.5

# Plot before training
y_pred = expit(a_t*x_tr+b_t)
MSE = np.sum((y_tr-y_pred)**2)/(N+1)
plt.title(f'Epoch=0 a={a_t:.2f} b={b_t:.2f} MSE={MSE:.2f}')
#plt.plot(x_h,y_h,'co', label="hobitti")
#plt.plot(x_e,y_e,'mo', label="haltija")
if gaud_sty.color==True:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
    x = np.linspace(0.0,+5.0,50)
    plt.plot(x,expit(a_t*x+b_t),'b-')
else:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    x = np.linspace(0.0,+5.0,50)
    plt.plot(x,expit(a_t*x+b_t),'k-')
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-0.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_1.png')
plt.show()

for e in range(1,num_of_epochs):
    grad_a = np.sum(2*x_tr*(y_tr-expit(a_t*x_tr+b_t))*expit(a_t*x_tr+b_t)*(-1+expit(a_t*x_tr+b_t)))
    grad_b = np.sum(2*(y_tr-expit(a_t*x_tr+b_t))*expit(a_t*x_tr+b_t)*(-1+expit(a_t*x_tr+b_t)))
    a_t = a_t-learning_rate*grad_a
    b_t = b_t-learning_rate*grad_b
    if np.mod(e,20) == 0 or e == 1:
        # Compute train error
        y_pred = expit(a_t*x_tr+b_t)
        MSE = np.sum((y_tr-y_pred)**2)/(N+1)
        plt.title(f'Epoch={e} a={a_t:.2f} b={b_t:.2f} MSE={MSE:.2f}')
        #plt.plot(x_h,y_h,'co', label="hobitti")
        #plt.plot(x_e,y_e,'mo', label="haltija")
        if gaud_sty.color==True:
            plt.plot(x_h,y_h, linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
            plt.plot(x_e,y_e,  linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
            x = np.linspace(0.0,+5.0,50)
            plt.plot(x,expit(a_t*x+b_t),'b-')
        else:
            plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
            plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
            x = np.linspace(0.0,+5.0,50)
            plt.plot(x,expit(a_t*x+b_t),'k-')
        plt.xlabel('pituus [m]')
        plt.axis([0.5,5.0,-0.1,+1.1])
        if e == 1:
            if gaud_sty.savefig:
                if gaud_sty.color:
                    plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_2_col.png')
                else:
                    plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_2.png')
        if e == 20:
            if gaud_sty.savefig:
                if gaud_sty.color:
                    plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_3_col.png')
                else:
                    plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_3.png')
        plt.show()

# Plot after training
y_pred = expit(a_t*x_tr+b_t)
MSE = np.sum((y_tr-y_pred)**2)/(N+1)
plt.title(f'Epoch={num_of_epochs} a={a_t:.2f} b={b_t:.2f} MSE={MSE:.2f}')
#plt.plot(x_h,y_h,'co', label="hobitti")
#plt.plot(x_e,y_e,'mo', label="haltija")
if gaud_sty.color==True:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
    x = np.linspace(0.0,+5.0,50)
    plt.plot(x,expit(a_t*x+b_t),'b-')
else:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    x = np.linspace(0.0,+5.0,50)
    plt.plot(x,expit(a_t*x+b_t),'k-')
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-0.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_4.png')
plt.show()

a = a_t
b = b_t

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
    x = np.linspace(0.0,+5.0,50)
    plt.plot(x,expit(a_t*x+b_t),'b-')
else:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    x = np.linspace(0.0,+5.0,50)
    plt.plot(x,expit(a_t*x+b_t),'k-')
#plt.plot(x_e_test,np.zeros([N_t,1]),'kv', label="tuntematon")
#plt.plot(x_h_test,np.zeros([N_t,1]),'kv')
if gaud_sty.color==True:
    plt.plot(x_e_test,np.zeros([N_t,1]), label="tuntematon", linestyle='', c='orange', marker=gaud_sty.marker3[2])
    plt.plot(x_h_test,np.zeros([N_t,1]), linestyle='', c='orange', marker=gaud_sty.marker3[2])
else:
    plt.plot(x_e_test,np.zeros([N_t,1]), label="tuntematon", linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[2])
    plt.plot(x_h_test,np.zeros([N_t,1]), linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[2])
plt.title('Luokiteltavia näytteitä')
plt.legend()
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-0.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_5_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_5.png')
plt.show()

#
# 5. Do classification and plot results
x_te = np.concatenate((x_h_test,x_e_test))
y_te = np.concatenate((1*np.ones([N_t,1]),2*np.ones([N_t,1])))
y_te_pred = expit(a*x_te+b)
#plt.plot(x_h,y_h,'co', label="hobitti")
#plt.plot(x_e,y_e,'mo', label="haltija")
if gaud_sty.color==True:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c='black', marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c='red', marker=gaud_sty.marker3[1])
    plt.legend()
    plt.plot(x,expit(a*x+b),'b-')
else:
    plt.plot(x_h,y_h, linestyle='', label="maahinen", c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x_e,y_e,  linestyle='', label="haltija", c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    plt.legend()
    plt.plot(x,expit(a*x+b),'k-')

corr_class = 2*N_t
for s_ind, s in enumerate(x_te):
    print(y_te[s_ind])
    if y_te_pred[s_ind] > 0.5:
        #plt.plot(s,1,'mv')
        if gaud_sty.color==True:
            plt.plot(s,1,  linestyle='', c='orange', marker=gaud_sty.marker3[1])
        else:
            plt.plot(s,1,  linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[1])
        if y_te[s_ind] != 2: # wrong class
            plt.plot(s,1, 'kx', markersize=30)
            corr_class = corr_class-1
    else:
        #plt.plot(s,0, 'cv')
        if gaud_sty.color==True:
            plt.plot(s,0,  linestyle='', c='orange', marker=gaud_sty.marker3[0])
        else:
            plt.plot(s,0,  linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[0])
        if y_te[s_ind] != 1: # wrong class
            plt.plot(s,0, 'kx', markersize=30)
            corr_class = corr_class-1
plt.title('Luokittelutulos (suoran sovitus)')
#plt.axis([0.5,2.5,-1.1,+1.1])
tot_correct = corr_class/(2*N_t)
print(f'Classication accuracy: {tot_correct*100:.2f}%')
plt.xlabel('pituus [m]')
plt.axis([0.5,5.0,-0.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_6_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch03_line_fitting_sigmoidal_6.png')
plt.show()

