#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:18:51 2019

@author: tbarton
"""

# Time series encoder 
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.manifold import TSNE
from keras.models import model_from_yaml
from sklearn.cluster import KMeans
import numpy as np
from metric_functions import *

def wf_autoencoder(xtrain, xtest = None, encoding_dim = 1000, epochs = 50, batch_size = 300, validation_data = None, validation_split = .1, load_model = None, save_model=False):
    """
    

    Parameters
    ----------
    xtrain : nd.array
        Data to encode.
    xtest : nd.array, optional
        Data that you wish to use for verification. This parameter is rarely advised. The default is None.
    encoding_dim : int, optional
        Desired output dimension. The default is 1000.
    epochs : int, optional
        How many itterations to run the data through in the neural net. The default is 50.
    batch_size : int, optional
        How to divide the data per epoch. 'Batch' data points will run through the model itteratively. The default is 300.
    validation_data : Bool, optional
        weither to use the xtest data or not. The default is None.
    validation_split : int, optional
        If no validation data, what portion of the data should be set aside for validation. The default is .1.
    load_model : list, optional
        If not None, then a model will be attempted to be loaded from the current working directory. the structure of this
        input should be as follows [[enoding_model_name_ymal, encoding_model_wights_h5], [autoenoding_model_name_ymal, autoencoding_model_wights_h5]].
        These are the NAMES of the files, not the files themselves. do NOT include the suffix of these files. Enter 'Standard' if the filies are of the 
        standard naming convention outputted by 'save_model = True'. The default is None.
    save_model : Bool, optional
        If not None, the model will be run and saved. This will be overridden by load_model. The default is False.

    Returns
    -------
    A length 3 list.
    
    1. encoding model.
    2. full autoencoding model.
    3. the encoded xtrain waves.

    """
    if load_model is None:
        input_wave = Input(shape=(xtrain.shape[1],))   
        encoded1 = Dense(9000, activation = 'sigmoid')(input_wave)  
        encoded2 = Dense(9000, activation = 'relu')(encoded1)  
        encoded3 = Dense(7000, activation = 'relu', activity_regularizer=keras.regularizers.l2(0.01))(encoded2)   
        encoded4 = Dense(5000, activation = 'relu', activity_regularizer=keras.regularizers.l1(0.01))(encoded3)  
        encoded5 = Dense(3000, activation = 'relu', )(encoded4)  
        encoded = Dense(encoding_dim, activation = 'relu')(encoded5)  
        decoded2 = Dense(3000, activation = 'relu')(encoded)   
        decoded3 = Dense(5000, activation = 'relu')(decoded2)  
        decoded4 = Dense(7000, activation = 'relu')(decoded3)   
        decoded5 = Dense(9000, activation = 'relu')(decoded4)  
        decoded6 = Dense(9000, activation = 'relu')(decoded5)  
        decoded = Dense(xtrain.shape[1], activation = 'sigmoid')(decoded6)
          
        autoencoder = Model(input_wave, decoded)
        autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
        if validation_data is None:
            autoencoder.fit(xtrain, xtrain, 
                            epochs = epochs, batch_size = batch_size,
                            shuffle = True, 
                            validation_split = validation_split)
        else:
            autoencoder.fit(xtrain, xtrain, 
                            epochs = epochs, batch_size = batch_size,
                            shuffle = True, 
                            validation_data = [xtest, xtest])
        
        encode = Model(input_wave, encoded)
        
        encoded_input = Input(shape = (encoding_dim,))
    #    decoder_layer = autoencoder.layers[-1]
    #    decode = Model(encoded_input, decoded)
        encoded_waves = encode.predict(xtrain)
    #    decoded_waves = decode.predict(encoded_waves)
        
        
        if save_model == True:
            encode_yaml = encode.to_yaml()
            with open("encode_yaml.yaml", "w") as yaml_file:
                yaml_file.write(encode_yaml)
            # serialize weights to HDF5
            encode.save_weights("encode.h5")
            
            autoencoder_yaml = autoencoder.to_yaml()
            with open("autoencoder_yaml.yaml", "w") as yaml_file:
                yaml_file.write(autoencoder_yaml)
            # serialize weights to HDF5
            autoencoder.save_weights("autoencoder.h5")
            print("Saved model to disk")    
        
        
        return([encode, autoencoder, encoded_waves])

    else:
        if load_model == 'Standard':
            load_model = [['encode', 'encode'],['autoencoder', 'autoencoder']]
        yaml_file = open('{}.yaml'.format(load_model[0][0]), 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        encode = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        encode.load_weights("{}.h5".format(load_model[0][1]))
        print("Loaded encoding from disk")
        
        yaml_file = open('{}.yaml'.format(load_model[1][0]), 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        autoencoder = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        autoencoder.load_weights("{}.h5".format(load_model[1][1]))
        print("Loaded autoencoding from disk")
        
        
        encoded_waves = encode.predict(xtrain)
        full_waves = autoencoder.predict(xtrain)
        return([encoded_waves, full_waves])
#################### Testing time #######################
    
#xtrain = dat_val.fillna(0).apply(normalize, 1).apply(pd.Series)
#encoding_dim = 1000
#temp = wf_autoencoder(xtrain, epochs = 1)


test = wf_autoencoder(normalize(i_data.iloc[:,:-1]), load_model = 'Standard')

test[0] = pd.DataFrame(test[0])

'''
When you come back
    1. Get Kmeans clustering going for 
        a. raw data
        b. sampled down data
        c. autoencoded data 
'''


# 1. Kmeans on red


teskm = KMeans(n_clusters = 9, n_jobs = 4)

teskm.fit(test[0])
testkm = teskm.predict(test[0].dropna())
np.argmax(testkm, 1)
np.unique(testkm[0])

# 2. Kmeans on full

teskm = KMeans(n_clusters = 9, n_jobs = 4)

teskm.fit(pd.DataFrame(normalize(i_data.iloc[:,:-1])))
testkm = teskm.predict(pd.DataFrame(test[0]).dropna())
testkm = Kmeans(i_data.iloc[:, :-2], len(np.unique(i_data['label'])))



plot(i_data.iloc[0, :-1])
plot(test[0].iloc[0, :])


