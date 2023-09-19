import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.linalg import inv
plt.style.use('book.mplstyle')
import gaud_sty

#
# Let's make 2D correlated data of hobbits and elfs

# Hobits
np.random.seed(42) # to always get the same points
N_h = 80 
x1_h = np.random.normal(1.1,0.3,N_h)
a_h = 50.0
b_h = 20.0
x2_h_noise =  np.random.normal(0,8,N_h)
x2_h = a_h*x1_h+b_h+x2_h_noise

# Elves
N_e = 20 
x1_e = np.random.normal(1.9,0.4,N_e)
a_e = 30.0
b_e = 30.0
x2_e_noise =  np.random.normal(0,8,N_e)
x2_e = a_e*x1_e+b_e+x2_e_noise

#
# PLOT 1 both classes in 1D
if gaud_sty.color==True:
    plt.plot(x1_h,np.zeros([N_h,1]), label="maahinen", linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x1_e,np.zeros([N_e,1]), label="haltija", linestyle='', marker=gaud_sty.marker3[1])
else:
    plt.plot(x1_h,np.zeros([N_h,1]), label="maahinen", linestyle='', c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x1_e,np.zeros([N_e,1]), label="haltija", linestyle='', c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
plt.title('Opetusnäytteitä kahdesta luokasta c1 ja c2')
plt.legend()
plt.xlabel('pituus [m]')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_1.png')
plt.show()

if gaud_sty.color==True:
    plt.hist(x1_h, bins = 10, alpha=0.5)
    plt.hist(x1_e, bins = 10, alpha=0.5)
else:
    plt.hist(x1_h, bins = 10, alpha=0.5, color=gaud_sty.gray2[1])
    plt.hist(x1_e, bins = 10, alpha=0.5, color=gaud_sty.gray2[0])
plt.title('Opetusnäytteitä kahdesta luokasta c1 ja c2')
plt.xlabel('pituus [m]')
plt.ylabel('lukumäärä [kpl]')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_2.png')
plt.show()


#
# PLOT 2 both classes in 1D
if gaud_sty.color==True:
    plt.plot(x2_h,np.zeros([N_h,1]), label="maahinen", linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x2_e,np.zeros([N_e,1]), label="haltija", linestyle='', marker=gaud_sty.marker3[1])
else:
    plt.plot(x2_h,np.zeros([N_h,1]), label="maahinen", linestyle='', c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x2_e,np.zeros([N_e,1]), label="haltija", linestyle='', c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
plt.title('Opetusnäytteitä kahdesta luokasta c1 ja c2')
plt.legend()
plt.xlabel('paino [kg]')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_3.png')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_3.png')
plt.show()

if gaud_sty.color==True:
    plt.hist(x2_h, bins = 10, alpha=0.5)
    plt.hist(x2_e, bins = 10, alpha=0.5)
else:
    plt.hist(x2_h, bins = 10, alpha=0.5, color=gaud_sty.gray2[1])
    plt.hist(x2_e, bins = 10, alpha=0.5, color=gaud_sty.gray2[0])
plt.title('Opetusnäytteitä kahdesta luokasta c1 ja c2')
plt.xlabel('paino [kg]')
plt.ylabel('lukumäärä [kpl]')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_4.png')
plt.show()

#
# PLOT 2 both classes in 2D
if gaud_sty.color==True:
    plt.plot(x1_h,x2_h, label="maahinen", linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x1_e,x2_e, label="haltija", linestyle='', marker=gaud_sty.marker3[1])
else:
    plt.plot(x1_h,x2_h, label="maahinen", linestyle='', c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x1_e,x2_e, label="haltija", linestyle='', c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
plt.title('Opetusnäytteitä kahdesta luokasta c1 ja c2')
plt.legend()
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
#plt.axis([0.5,2.5,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_5_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_5.png')
plt.show()

#
# Let's make test data and test

N_h_test = 10
x1_h_test = np.random.normal(1.1,0.3,N_h_test)
x2_h_test_noise =  np.random.normal(0,8,N_h_test)
x2_h_test = a_h*x1_h_test+b_h+x2_h_test_noise
c_h_test = np.zeros(N_h_test)
c_h_test[:] = 1

N_e_test = 10
x1_e_test = np.random.normal(1.9,0.4,N_e_test)
x2_e_test_noise =  np.random.normal(0,8,N_e_test)
x2_e_test = a_e*x1_e_test+b_e+x2_e_test_noise
c_e_test = np.zeros(N_e_test)
c_e_test[:] = 2

if gaud_sty.color==True:
    plt.plot(x1_h,x2_h, label="maahinen", linestyle='', marker=gaud_sty.marker3[0])
    plt.plot(x1_e,x2_e, label="haltija", linestyle='', marker=gaud_sty.marker3[1])
    plt.plot(x1_h_test,x2_h_test,label="testi-maah.", linestyle='', marker=gaud_sty.marker3[0],markerfacecolor=gaud_sty.gray3[2])
    plt.plot(x1_e_test,x2_e_test, label="testi-halt.", linestyle='', marker=gaud_sty.marker3[1],markerfacecolor=gaud_sty.gray3[2])
else:
    plt.plot(x1_h,x2_h, label="maahinen", linestyle='', c=gaud_sty.gray3[0], marker=gaud_sty.marker3[0])
    plt.plot(x1_e,x2_e, label="haltija", linestyle='', c=gaud_sty.gray3[1], marker=gaud_sty.marker3[1])
    plt.plot(x1_h_test,x2_h_test,label="testi-maah.", linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[0],markerfacecolor=gaud_sty.gray3[2])
    plt.plot(x1_e_test,x2_e_test, label="testi-halt.", linestyle='', c=gaud_sty.gray3[2], marker=gaud_sty.marker3[1],markerfacecolor=gaud_sty.gray3[2])
plt.title('Opetus- ja testinäytteitä kahdesta luokasta c1 ja c2')
plt.legend(loc='lower right')
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
#plt.axis([0.5,2.5,-1.1,+1.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_6_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gaussian2d_6.png')
plt.show()



x1_test = np.concatenate((x1_h_test,x1_e_test))
x2_test = np.concatenate((x2_h_test,x2_e_test))
c_test = np.concatenate((c_h_test,c_e_test))
c_test_hat = np.zeros(c_test.shape)

priori_h = N_h/(N_h+N_e)
mu1_h = np.mean(x1_h)
mu2_h = np.mean(x2_h)
sigma2_1_h = np.var(x1_h)
sigma2_2_h = np.var(x2_h)
p_h_1 = 1/np.sqrt(2*np.pi*sigma2_1_h)*np.exp(-1/(2*sigma2_1_h)*(x1_test-mu1_h)**2)
P_h_1 = priori_h*p_h_1
p_h_2 = 1/np.sqrt(2*np.pi*sigma2_2_h)*np.exp(-1/(2*sigma2_2_h)*(x2_test-mu2_h)**2)
P_h_2 = priori_h*p_h_2
P_h = priori_h*p_h_1*p_h_2


priori_e = N_e/(N_h+N_e)
mu1_e = np.mean(x1_e)
mu2_e = np.mean(x2_e)
sigma2_1_e = np.var(x1_e)
sigma2_2_e = np.var(x2_e)
p_e_1 = 1/np.sqrt(2*np.pi*sigma2_1_e)*np.exp(-1/(2*sigma2_1_e)*(x1_test-mu1_e)**2)
P_e_1 = priori_e*p_e_1
p_e_2 = 1/np.sqrt(2*np.pi*sigma2_2_e)*np.exp(-1/(2*sigma2_2_e)*(x2_test-mu2_e)**2)
P_e_2 = priori_e*p_e_2
P_e = priori_e*p_e_1*p_e_2

c_test_hat[np.argwhere(np.greater(P_h_1,P_e_1) == True)] = 1
c_test_hat[np.argwhere(np.greater(P_h_1,P_e_1) == False)] = 2
corr = np.count_nonzero(np.equal(c_test,c_test_hat))
success_rate = corr/(N_h_test+N_e_test)
print(f'Success rate using height: {success_rate:.2f}')

c_test_hat[np.argwhere(np.greater(P_h_2,P_e_2) == True)] = 1
c_test_hat[np.argwhere(np.greater(P_h_2,P_e_2) == False)] = 2
corr = np.count_nonzero(np.equal(c_test,c_test_hat))
success_rate = corr/(N_h_test+N_e_test)
print(f'Success rate using weight: {success_rate:.2f}')

c_test_hat[np.argwhere(np.greater(P_h,P_e) == True)] = 1
c_test_hat[np.argwhere(np.greater(P_h,P_e) == False)] = 2
corr = np.count_nonzero(np.equal(c_test,c_test_hat))
success_rate = corr/(N_h_test+N_e_test)
print(f'Success rate using both (naively): {success_rate:.2f}')
