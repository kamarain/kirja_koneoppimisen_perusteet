import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras

# Lataa datajoukot
x_tr = np.loadtxt("x_train_medium.txt")
y_tr = np.loadtxt("y_train_medium.txt")
x_te = np.loadtxt("x_test_medium.txt")
y_te = np.loadtxt("y_test_medium.txt")

# Tee neuroverkko
model = Sequential()
# 1 tai 100
model.add(Dense(1, input_dim=1, activation='sigmoid'))
# Ulostuloja aina kaksi, yksi kummallekin luokalle
model.add(Dense(2, activation='sigmoid'))
# 0.1 tai 0.001
opt = keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=opt, loss='mse', metrics=['mse'])

# Yksi-kuuma (one hot) -koodataan luokka 1 -> [1 0] 2 -> [0 1]
y_tr_2 = np.empty([y_tr.shape[0],2])
y_tr_2[np.where(y_tr==1),0] = 1
y_tr_2[np.where(y_tr==1),1] = 0
y_tr_2[np.where(y_tr==2),0] = 0
y_tr_2[np.where(y_tr==2),1] = 1

# Opetus - epokkeja 1 tai 100
model.fit(x_tr, y_tr_2, epochs=100, verbose=1)

# Tulokset opetuspisteille
y_tr_pred = np.empty(y_tr.shape)
y_tr_pred_2 = np.squeeze(model.predict(x_tr))
for pred_ind in range(y_tr_pred_2.shape[0]):
    if y_tr_pred_2[pred_ind][0] > y_tr_pred_2[pred_ind][1]:
        y_tr_pred[pred_ind] = 1
    else:
        y_tr_pred[pred_ind] = 2

tot_correct = len(np.where(y_tr-y_tr_pred == 0)[0])
print(f'Classication accuracy (training data): {tot_correct/len(y_tr)*100}%')

# Tulokset testauspisteille
y_te_pred = np.empty(y_te.shape)
y_te_pred_2 = np.squeeze(model.predict(x_te))
for pred_ind in range(y_te_pred_2.shape[0]):
    if y_te_pred_2[pred_ind][0] > y_te_pred_2[pred_ind][1]:
        y_te_pred[pred_ind] = 1
    else:
        y_te_pred[pred_ind] = 2

tot_correct = len(np.where(y_te-y_te_pred == 0)[0])
print(f'Classication accuracy (test data): {tot_correct/len(y_te)*100}%')
