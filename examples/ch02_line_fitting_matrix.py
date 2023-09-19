import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.linalg import inv
plt.style.use('book.mplstyle')
import gaud_sty

#
# 1. Generate and plot the same N random points as in ch02_line_fitting.py
np.random.seed(42) # to always get the same points
N = 50 
x = np.random.normal(1.1,0.3,N)
a_gt = 50.0
b_gt = 20.0
y_noise =  np.random.normal(0,8,N) # Measurements from the class 1\n",
y = a_gt*x+b_gt+y_noise
plt.plot(x,y,'ko')
plt.title('Opetusaineisto suoran sovitukseen')
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])
if gaud_sty.savefig:
    plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_matrix_1.png')
plt.show()

#
# 2. Do and plot LSQfit using the matrix form

# Form X and y
X = np.concatenate((np.transpose([np.ones(N)]),np.transpose([x])), axis=1)
w_foo = np.matmul(np.transpose(X),X)
w_foo = inv(w_foo)
w = np.matmul(np.matmul(y,X),w_foo)

a = w[1]
b = w[0]
y_h = a*x+b
MSE = np.sum((y-y_h)**2)/N

# Coodinate system
plt.plot(x,y,'ko')
x_plot = np.linspace(0,2.0,10)
if gaud_sty.color==True:
    plt.plot(x_plot,a*x_plot+b,'r-')
else:
    plt.plot(x_plot,a*x_plot+b,'k-')
plt.title(f"Sovitettu suora (a={a:.1f}, b={b:.1f}, MSE={MSE:.1f})")
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_matrix_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_matrix_2.png')
plt.show()

#
# 3. Do the Matrix LSQfit for 2nd order polynomial

# Second order polynomial
x = np.linspace(-1.0,+1.0,N)
w0 = -5
w1 = 2
w2 = 5
y_noise =  np.random.normal(0,0.25,N)
y=w0+w1*x+w2*x**2+y_noise
plt.plot(x,y,'ko')
plt.title(f"Kohinaisia pisteitä käyrältä y=5x^2+2x-5")
#plt.axis([-1,+1,-11,-7])
if gaud_sty.savefig:
    plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_matrix_3.png')
plt.show()

# Form X and y
X = np.concatenate((np.transpose([np.ones(N)]),np.transpose([x]), np.transpose([x**2])), axis=1)
w_foo = np.matmul(np.transpose(X),X)
w_foo = inv(w_foo)
w = np.squeeze(np.matmul(np.matmul(y,X),w_foo))
y_h = w[0]+w[1]*x+w[2]*x**2
MSE = np.sum((y-y_h)**2)/N
plt.plot(x,y,'ko')
if gaud_sty.color==True:
    plt.plot(x,y_h,'r-')
else:
    plt.plot(x,y_h,'k-')
plt.title(f"Sovitettu suora (w0={w[0]:.2f}, w1={w[1]:.2f}, w2={w[2]:.2f}, MSE={MSE:.2f})")
#plt.axis([-1,+1,-11,-7])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_matrix_4_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch02_line_fitting_matrix_4.png')
plt.show()
