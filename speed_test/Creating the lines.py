#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:18:04 2019

@author: tbarton
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import keras
import random as rd



bib = open('TheBible.txt', 'r').read()
bib = bib.lower()
chars = sorted(list(set(bib)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(bib)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 300
datax = []
datay = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = bib[i:i+seq_length]
    seq_out = bib[i+seq_length]
    datax.append([char_to_int[j] for j in seq_in])
    datay.append(char_to_int[seq_out])
print('{} total training points'.format(len(datax)))

X = np.reshape(datax, (len(datax), 300, 1))

np.array(datax).shape

X = X/float(n_vocab)
y = np_utils.to_categorical(datay)

y_new =[]
for i in range(len(y)):
   y_new.append([int(j) for j in y[i]])

y = np.array(y_new)


print('model being created')

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'tanh'))
model.add(Dense(y.shape[1], activation = 'softmax'))





model.compile(loss = 'categorical_crossentropy', 
              optimizer=keras.optimizers.adam())


model.load_weights('best_weights.hdf5')



def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))



new_vec = datax[rd.randint(0, len(datax))]
seed_vec = new_vec.copy()
for i in range(300):
    pred = np.argmax(model.predict(np.reshape([i/float(n_vocab) for i in new_vec], (1, 300, 1))))
    new_vec.append(pred)
    new_vec = new_vec[1:]
    
int_to_char = dict((i, c) for i, c in enumerate(chars))


verse = ''
for i in new_vec:
    verse = verse + int_to_char[i]
og = ''
for i in seed_vec:
    og = og + int_to_char[i]

print(og + '\n\n\n')
print(verse)
