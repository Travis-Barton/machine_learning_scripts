#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:23:13 2019

@author: tbarton
"""



from keras import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import Dense
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_catagorical
from scipy import signal



def cnn_classification_tool_1d(wf, classes, epochs, batch_size, optmizer = 'adam', loss = 'binary_crossentropy',  pool_size = 65, window_size = 130, val_split = .1):
    
    
    model = Sequential()
    model.add(Dense(wf.shape[1], input_shape =(wf.shape[1], 1), activation = 'tanh'))
    model.add(Conv1D(np.ceiling(wf.shape[1]**.5), window_size, activation = 'relu'))
    model.add(MaxPooling1D(pool_size))
    model.add(Conv1D(np.ceiling(wf.shape[1]**.5), window_size, activation = 'relu'))
    model.add(MaxPooling1D(pool_size))
    model.add(Dense(500, activation = 'relu'))
    model.add(Dense(len(np.unique(classes)), activation = 'softmax'))
    
    
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = ['acc'])
    
    classes = to_catagorical(classes)
    model.fit(wf, classes,
              epochs = epochs, 
              batch_size = batch_size,
              validation_split = val_split)
    
    return(model)



def cnn_classification_tool_2d(wf_pic, classes, epochs, batch_size, windowsize = (8, 8), optmizer = 'adam', loss = 'binary_crossentropy',  pool_size = (8, 8), window_size = 130, val_split = .1):
    
    
    model = Sequential()
    model.add(Conv2D(64, windowsize, input_shape=(10020, 10020, 1), padding = 'same', activation = 'tanh'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(64, windowsize, activation = 'relu'))
    model.add(MaxPooling2D(poolsize=pool_size))
    model.add(Dropout(.4))
    model.add(Conv2D(64, windowsize, activation = 'relu'))
    model.add(MaxPooling2D(poolsize=pool_size))
    model.add(Dropout(.2))
    model.add(Flatten())
    model.add(Dense(10020, activation = 'relu'), activity_regularizer=regularizers.l2(0.01))
    model.add(Dense(100, activation = 'sigmoid'), activity_regularizer=regularizers.l1(0.01))
    model.add(Dense(len(np.unique(classes)), activation = 'softmax'))
    
    
    
    
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = ['acc'])
    
    classes = to_catagorical(classes)
    model.fit(wf_pic, classes,
              epochs = epochs, 
              batch_size = batch_size,
              validation_split = val_split)
    
    return(model)




# model = Sequential()
# model.add(Conv2D(64, (10, 10), input_shape=(10020, 10020, 1), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(8, 8)))
# model.add(Conv2D(64, (10, 10), activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(8, 8)))
# model.add(Conv2D(64, (10, 10), activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(8, 8)))

# model.add(Dropout(.4))
# model.add(Flatten())
# model.add(Dense(10020, activation = 'relu'))
# model.add(Dense(100, activation = 'sigmoid'))
# model.add(Dense(2, activation = 'softmax'))
# model.summary()




##################################### Spectrogram tool #############################
def convert_to_spectrogram(wf):
    f, t, Sxx = signal.spectrogram(i_data.iloc[0,:10020], 130)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    
plot(i_data.iloc[1000,:10020])
convert_to_spectrogram(i_data.iloc[0,:10020])


fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

plot(x)


import pandas as pd
i_data = pd.read_csv('/Users/tbarton/Documents/GitHub/Personal-Projects/wave.csv')
f, t, Sxx = signal.spectrogram(i_data.values[:, 1], 130)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

