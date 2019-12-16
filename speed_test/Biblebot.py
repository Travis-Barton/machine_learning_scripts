

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import keras

print('imports done')

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

filepath = "best_weights2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
j=1
for i in np.repeat(4, 13):
    model.fit(X, y, epochs=i, batch_size=500, callbacks=callbacks_list)
    print('\n \n starting {} \n \n'.format(j))
    j+=1