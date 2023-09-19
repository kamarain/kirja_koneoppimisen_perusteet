import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.linalg import inv
from mpl_toolkits import mplot3d
plt.style.use('book.mplstyle')
import gaud_sty

from sklearn.mixture import GaussianMixture

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

# Dwarfs
N_d = 60 
x1_d = np.random.normal(1.0,0.1,N_d)
a_d = 50.0
b_d = 40.0
x2_d_noise =  np.random.normal(0,6,N_d)
x2_d = a_d*x1_d+b_d+x2_d_noise

print('Mean and covariance for hobits')
X_h = np.concatenate(([x1_h],[x2_h]),axis=0)
mu_h = np.mean(X_h,axis=1)
print(mu_h)
Sigma_h = np.cov(X_h)
print(Sigma_h)
priori_h = N_h/(N_h+N_e+N_d)


print('Mean and covariance for elfs')
X_e = np.concatenate(([x1_e],[x2_e]),axis=0)
mu_e = np.mean(X_e,axis=1)
print(mu_e)
Sigma_e = np.cov(X_e)
print(Sigma_e)
priori_e = N_e/(N_h+N_e+N_d)

print('Mean and covariance for dwarfs')
X_d = np.concatenate(([x1_d],[x2_d]),axis=0)
mu_d = np.mean(X_d,axis=1)
print(mu_d)
Sigma_d = np.cov(X_d)
print(Sigma_d)
priori_d = N_d/(N_h+N_e+N_d)

print('Mean and covariance for dwarfs and elves combined')
X_ed = np.concatenate((X_d, X_e),axis=1)
mu_ed = np.mean(X_ed,axis=1)
print(mu_ed)
Sigma_ed = np.cov(X_ed)
print(Sigma_ed)
priori_ed = (N_d+N_e)/(N_h+N_e+N_d)

def gaussian2d(X,Y, mu, Sigma, priori):
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = priori*1/np.sqrt(2*np.pi)*1/np.sqrt(np.linalg.det(Sigma))*np.exp(-1/2*np.transpose([X[i,j]-mu[0], Y[i,j]-mu[1]]) @ inv(Sigma) @ [X[i,j]-mu[0], Y[i,j]-mu[1]])
    return Z

x_1 = np.linspace(0,3.0,70)
x_2 = np.linspace(0,150.0,70)

X, Y = np.meshgrid(x_1,x_2)
Z_h = gaussian2d(X,Y, mu_h, Sigma_h, priori_h)
Z_e = gaussian2d(X,Y, mu_e, Sigma_e, priori_e)
Z_d = gaussian2d(X,Y, mu_d, Sigma_d, priori_d)
Z_ed = gaussian2d(X,Y, mu_ed, Sigma_ed, priori_ed)

fig = plt.figure()
ax = plt.axes(projection='3d')
if gaud_sty.color==True:
    ax.contour3D(X, Y, Z_h, 30, cmap='Blues', linewidths=1)
    ax.text(mu_h[0]+0.5,mu_h[1],0.10,'maahiset')
    ax.contour3D(X, Y, Z_e, 5, cmap='Reds', linewidths=4)
    ax.text(mu_e[0],mu_e[1],0.05,'haltiat')
    ax.contour3D(X, Y, Z_d, 50, cmap='Greens', linewidths=2)
    ax.text(mu_d[0],mu_d[1],0.25,'kääpiöt')
else:
    ax.contour3D(X, Y, Z_h, 30, cmap='gray', linewidths=1)
    ax.text(mu_h[0]+0.5,mu_h[1],0.10,'maahiset')
    ax.contour3D(X, Y, Z_e, 5, cmap='gray', linewidths=4)
    ax.text(mu_e[0],mu_e[1],0.05,'haltiat')
    ax.contour3D(X, Y, Z_d, 50, cmap='gray', linewidths=2)
    ax.text(mu_d[0],mu_d[1],0.25,'kääpiöt')
ax.set_xlabel('pituus [m]')
ax.set_ylabel('paino [kg]')
ax.set_zlabel('Todennäköisyystiheys');
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gmm_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gmm_1.png')
#plt.legend()
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
if gaud_sty.color==True:
    ax.contour3D(X, Y, Z_h, 50, cmap='Blues', linewidths=1, alpha=0.5)
    ax.text(mu_h[0],mu_h[1],0.10,'maahiset')
    ax.contour3D(X, Y, Z_ed, 20, cmap='Oranges', linewidths=3)
    ax.text(1.1,150,0.02,'haltiat+kääpiöt')
    ax.set_xlabel('pituus [m]')
else:
    ax.contour3D(X, Y, Z_h, 50, cmap='gray', linewidths=1, alpha=0.5)
    ax.text(mu_h[0],mu_h[1],0.10,'maahiset')
    ax.contour3D(X, Y, Z_ed, 20, cmap='gray', linewidths=3)
    ax.text(1.1,150,0.02,'haltiat+kääpiöt')
    ax.set_xlabel('pituus [m]')
ax.set_ylabel('paino [kg]')
ax.set_zlabel('Todennäköisyystiheys');
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gmm_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gmm_2.png')
#plt.legend()
plt.show()

#
# Run Gaussian mixture model
gm = GaussianMixture(n_components=2, random_state=0).fit(X_ed.T)
mu_ed1 = gm.means_[0]
mu_ed2 = gm.means_[1]
Sigma_ed1 = gm.covariances_[0]
Sigma_ed2 = gm.covariances_[0]
priori_ed1 = gm.weights_[0]/(gm.weights_[0]+gm.weights_[1])*priori_ed
priori_ed2 = gm.weights_[1]/(gm.weights_[0]+gm.weights_[1])*priori_ed
Z_ed_1 = gaussian2d(X,Y, mu_ed1, Sigma_ed1, priori_ed1)
Z_ed_2 = gaussian2d(X,Y, mu_ed2, Sigma_ed2, priori_ed2)
print('FOO')
print(mu_ed1)
print('FOO')

fig = plt.figure()
ax = plt.axes(projection='3d')
if gaud_sty.color==True:
    ax.contour3D(X, Y, Z_h, 50, cmap='Blues', linewidths=1, alpha=0.5)
    ax.text(mu_h[0]-0.9,mu_h[1]-75,0.10,'maahiset')
    ax.contour3D(X, Y, Z_ed_1,  30, cmap='Oranges', linewidths=1)
    ax.contour3D(X, Y, Z_ed_2,  20, cmap='Oranges', linewidths=1)
else:
    ax.contour3D(X, Y, Z_h, 50, cmap='gray', linewidths=1, alpha=0.5)
    ax.text(mu_h[0]-0.9,mu_h[1]-75,0.10,'maahiset')
    ax.contour3D(X, Y, Z_ed_1,  30, cmap='gray', linewidths=1)
    ax.contour3D(X, Y, Z_ed_2,  20, cmap='gray', linewidths=1)
ax.text(mu_e[0]-0.3,mu_e[1],0.09,'haltiat+kääpiöt 1')
ax.text(mu_d[0],mu_d[1],0.25,'haltiat+kääpiöt 2')
ax.set_xlabel('pituus [m]')
ax.set_ylabel('paino [kg]')
ax.set_zlabel('Todennäköisyystiheys');
plt.legend()
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch05_gmm_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch05_gmm_3.png')
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

x1_test = np.concatenate((x1_h_test,x1_e_test))
x2_test = np.concatenate((x2_h_test,x2_e_test))
c_test = np.concatenate((c_h_test,c_e_test))
c_test_hat = np.zeros(c_test.shape)

P_h = np.zeros(x1_test.shape)
P_e = np.zeros(x1_test.shape)
P_ed = np.zeros(x1_test.shape)
for i in range(x1_test.shape[0]):
    P_h[i] = priori_h*1/np.sqrt(2*np.pi)*1/np.sqrt(np.linalg.det(Sigma_h))*np.exp(-1/2*np.transpose([x1_test[i]-mu_h[0], x2_test[i]-mu_h[1]]) @ inv(Sigma_h) @ [x1_test[i]-mu_h[0], x2_test[i]-mu_h[1]])
    P_e[i] = priori_e*1/np.sqrt(2*np.pi)*1/np.sqrt(np.linalg.det(Sigma_e))*np.exp(-1/2*np.transpose([x1_test[i]-mu_e[0], x2_test[i]-mu_e[1]]) @ inv(Sigma_e) @ [x1_test[i]-mu_e[0], x2_test[i]-mu_e[1]])
    P_ed[i] = priori_ed*1/np.sqrt(2*np.pi)*1/np.sqrt(np.linalg.det(Sigma_ed))*np.exp(-1/2*np.transpose([x1_test[i]-mu_ed[0], x2_test[i]-mu_ed[1]]) @ inv(Sigma_ed) @ [x1_test[i]-mu_ed[0], x2_test[i]-mu_ed[1]])

print(P_h)
print(P_e)
print(P_ed)
print(np.greater(P_h,P_ed))
c_test_hat[np.argwhere(np.greater(P_h,P_ed) == True)] = 1
c_test_hat[np.argwhere(np.greater(P_h,P_ed) == False)] = 2
corr = np.count_nonzero(np.equal(c_test,c_test_hat))
success_rate = corr/(N_h_test+N_e_test)
print(f'Success rate using mixed: {success_rate:.2f}')
