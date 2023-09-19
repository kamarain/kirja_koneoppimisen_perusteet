# Tarvittavat kirjastot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras

# Sequential mahdollistaa kerrosten lisaamisen yksi kerrallaan
model = Sequential()
# Lisataan yksi kerros ja sille yksi paattelin
model.add(Dense(1, input_dim=1, activation='linear'))
# Oppimisnopeus
keras.optimizers.SGD(lr=0.5)
# SGD on perusmenetelma optimointiin ja virhe (loss), jota minimoidaan on MSE
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
