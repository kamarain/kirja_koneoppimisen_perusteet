import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Coodinate system
plt.xlabel('pituus [m]')
#plt.ylabel('paino [kg]')
plt.axis([0.5,2.5,-1.1,+1.1])

# Generate random points for training
np.random.seed(11) # to always get the same points
N = 200
x_h = np.random.normal(1.1,0.3,N)
x_e = np.random.normal(1.9,0.4,N)
plt.plot(x_h,np.zeros([N,1]),'ro', markersize=8, label="hobitti")
plt.plot(x_e,np.zeros([N,1]),'gs', markersize=8, label="haltija")
plt.title('Näytteitä kahdesta luokasta')
plt.legend()
plt.show()

# Generate random points for testing
N_t =  50
x_h_test = np.random.normal(1.1,0.3,N_t) # h as hobit
x_e_test = np.random.normal(1.9,0.4,N_t) # e as elf
plt.plot(x_h,np.zeros([N,1]),'ro', markersize=8, label="hobitti")
plt.plot(x_e,np.zeros([N,1]),'gs', markersize=8, label="haltija")
plt.plot(x_e_test,np.zeros([N_t,1]),'kv',linewidth=1, markersize=12, label="tuntematon")
plt.plot(x_h_test,np.zeros([N_t,1]),'kv', markersize=12)
plt.title('Luokiteltavia näytteitä')
plt.legend()
plt.show()

# 1-NN classifier

# Form the train input and output vectors  (1: hobit, 2: elf)
x_tr = np.concatenate((x_h,x_e))
y_tr = np.concatenate((1*np.ones([x_h.shape[0],1]),2*np.ones([x_e.shape[0],1])))

# Form the test input and output vectors
x_te = np.concatenate((x_h_test,x_e_test))
y_te = np.concatenate((1*np.ones([N_t,1]),2*np.ones([N_t,1])))

np.savetxt("x_train_medium.txt", x_tr)
np.savetxt("y_train_medium.txt", y_tr)
np.savetxt("x_test_medium.txt", x_te)
np.savetxt("y_test_medium.txt", y_te)
