import matplotlib.pyplot as plt
import numpy as np

import gaud_sty
plt.style.use('book.mplstyle')

# Coodinate system
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.axis([0,2,0,150])

# Generate random points
np.random.seed(42) # to always get the same points
N = 50 
x = np.random.normal(1.1,0.3,N)
a_gt = 50.0
b_gt = 20.0
y_noise =  np.random.normal(0,8,N) # Measurements from the class 1\n",
y = a_gt*x+b_gt+y_noise
if gaud_sty.color==True:
    plt.plot(x,y,'ro')
else:
    plt.plot(x,y,'ko')
plt.title('Opetusaineisto suoran sovitukseen')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch07_mlp_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch07_mlp_1.png')
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
# Model sequential
model = Sequential()
# 1st hidden layer (we also need to tell the input dimension)
#   10 neurons, but you can change to play a bit
model.add(Dense(1, input_dim=1, activation='linear'))
## 2nd hidden layer - YOU MAY TEST THIS
#model.add(Dense(10, activation='sigmoid'))
# Output layer
#model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='tanh'))
# Learning rate has huge effect 
keras.optimizers.SGD(lr=0.5)
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

num_of_epochs = 30
tr_hist = model.fit(x, y, epochs=num_of_epochs, verbose=1)

plt.plot(tr_hist.history['loss'],'k-')
plt.ylabel('häviö')
plt.xlabel('epokki')
#plt.legend(['opetus'], loc='upper right')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch07_mlp_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch07_mlp_2.png')
plt.show()


y_h = np.squeeze(model.predict(x))
MSE = np.sum((y-y_h)**2)/N
plt.xlabel('pituus [m]')
plt.ylabel('paino [kg]')
plt.title(f'Epoch={num_of_epochs} MSE={MSE:.2f}')
plt.axis([0,2,0,150])
if gaud_sty.color==True:
    plt.plot(x,y,'ro')
    plt.plot(x,y_h,'k-')
else:
    plt.plot(x,y,'ko')
    plt.plot(x,y_h,'k-')
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch07_mlp_3_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch07_mlp_3.png')
plt.show()
