#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:33:41 2019

@author: tbarton
"""
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.manifold import TSNE
from keras.models import model_from_yaml
#from Autoencoder_to_TSNE_Reduction import wf_autoencoder
import pandas as pd
import keras
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#import modin.padas as pd

import os
import sys

def codes_done(title = 'Hey Dude', msg = 'Your code is done'):
    os.system("osascript -e 'display notification \"{}\" with title \"{}\"'".format(msg, title))


def add_package_to_sys_path(base, package_relative_path):
    package_path = os.path.join(base, package_relative_path)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)


flag = 0
MAX_VAL = 9000000
CHUNKSIZE = 100000
WAVEFORM_LEN = 28000
EMBEDDING_DIM = 1000


PROJECTS_ROOT = os.path.realpath(os.path.join(os.getcwd(), '../..'))
PARENT_ROOT = os.path.realpath(os.path.join(os.getcwd(), '../../..'))



add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/ae/db')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/ae/common')

add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/analysis')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/analysis/inrush')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/analysis/i_wf_channel_switching')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/analysis/e_wf_features_wt')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/demo')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/db')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/test')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/tools')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/analysis/travis_analysis')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/analysis/jl_analysis')
add_package_to_sys_path(PROJECTS_ROOT, '/home/ampleadmin/Documents/advanced-projects/python/projects/wfl/analysis/i_wf_hiamp_peak_features')




import db_config as dbc
import get_wf_by_ids as gwi
import numpy as np
from detect_peaks import detect_peaks
from pq_utils import *
from feeder_view_plots import get_feeder_view_plot
import random as rd
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from statistics import mode
import os
import scipy.stats as stats
import random as rd
import time

def get_dist_wfs_by_dwr_table_id(wfl_conn, start_id, end_id):
    sql = """
        SELECT 
            dwr.id,
            b.disturbance_id, 
            dwr.e_wf_raw_bytes, 
            dwr.i_wf_raw_bytes,
            b.trigger_sample,
            b.e_post_offset, 
            b.e_post_scale, 
            b.i_post_offset, 
            b.i_post_scale,
            b.trigger_time_usec
        FROM disturbance_waveforms_raw dwr
        INNER JOIN disturbance_mm3_basics b
        on dwr.waveform_id = b.waveform_id
        WHERE 
        e_len = 10020
        AND
        i_len = 10020
    """
    sqlr = sql + """AND dwr.id >= {start_id} and dwr.id < {end_id} """.format(start_id=start_id, end_id=end_id)
    return pd.read_sql(sqlr, wfl_conn)


def normalize(vec):
    def calc(x, mi, ma):
        return((x-mi)/(ma-mi))
    mi = np.nanmin(vec)
    ma = np.nanmax(vec)
    return(np.apply_along_axis(calc, 0, vec, mi, ma))



def waveform_chopper(data, y = None, full = True, breaks = 75):
    '''
    description: 
        
    '''
    if full == True:
        ret = pd.DataFrame()
        i = 0
        times = [[],[],[]]
        for index, x in tqdm(data.iterrows(), total=data.shape[0]):
            ret_temp = pd.DataFrame()
            for j in range(breaks):
                s = time.time()
                
                red_row = x.iloc[np.r_[np.arange(j, data.shape[1], breaks)[0:-1]]].values
    #            red_row = red_row.reset_index()
                times[0].append(time.time()-s)
                s = time.time()
                if y is not None:
                    red_row[len(red_row)] = y.iloc[i]
                else:
                    red_row = red_row[:-1]
                times[1].append(time.time()-s)
                s = time.time()
                red_row = pd.Series(red_row)
                red_row.name = x.name
    #            ret = ret.reset_index(drop = True)
                ret_temp = ret_temp.append(red_row)            
            
                
                times[2].append(time.time()-s)
            ret = ret.append(ret_temp, ignore_index = True)
            i += 1
        ret = ret.rename({len(red_row):'label'}, axis = 'columns')
        print('\n Part 1: {} \n Part 2: {} \n Part 3: {}'.format(np.mean(times[0]), np.mean(times[1]), np.mean(times[2])))   
        return(ret)
    else:
        ret = pd.DataFrame()
        i = 0
        times = [[],[],[]]
        for index, x in tqdm(data.iterrows(), total=data.shape[0]):
            
            red_row = x.iloc[np.r_[np.arange(5, data.shape[1], breaks)[0:-1]]].values
            if y is not None:
                red_row[len(red_row)] = y.iloc[i]
            else:
                red_row = red_row[:-1]
            red_row = pd.Series(red_row)
            red_row.name = x.name
#            ret = ret.reset_index(drop = True)
        
            
            ret = ret.append(red_row, ignore_index = True)
            i += 1
        ret = ret.rename({len(red_row):'label'}, axis = 'columns')
        #print('\n Part 1: {} \n Part 2: {} \n Part 3: {}'.format(np.mean(times[0]), np.mean(times[1]), np.mean(times[2])))   
        return(ret)



input_wave = Input(shape=(9240,))   
encoded1 = Dense(9000, activation = 'sigmoid')(input_wave)  
encoded2 = Dense(7000, activation = 'sigmoid')(encoded1)   
encoded3 = Dense(5000, activation = 'sigmoid')(encoded2)   
encoded4 = Dense(3000, activation = 'sigmoid')(encoded3)  
encoded = Dense(EMBEDDING_DIM, activation = 'sigmoid')(encoded4)  
decoded3 = Dense(3000, activation = 'sigmoid')(encoded)  
decoded4 = Dense(5000, activation = 'sigmoid')(decoded3)   
decoded5 = Dense(7000, activation = 'sigmoid')(decoded4)  
decoded6 = Dense(9000, activation = 'sigmoid')(decoded5)  
decoded = Dense(9240, activation = 'sigmoid')(decoded6)
  
autoencoder = Model(input_wave, decoded)
autoencoder.compile(optimizer = 'sgd', loss = 'mean_squared_error')



for i in np.arange(0, MAX_VAL, CHUNKSIZE):    
    print('trying {}-{}'.format(i, i+CHUNKSIZE))
    temp_db = get_dist_wfs_by_dwr_table_id(dbc.db_connect(dbc.wfl_db_config), i, i+CHUNKSIZE)
    # temp = pd.DataFrame(temp_db['e_wf_raw_bytes'].apply(np.frombuffer, dtype = '<i4').apply(pd.Series))
    # temp = temp.multiply(temp_db['e_post_scale'], axis = 0)
    # e_data = e_data.append(temp)  
    
    try:
        # temp = pd.DataFrame(temp_db['i_wf_raw_bytes'].apply(np.frombuffer, dtype = '<i4').apply(pd.Series))
        # temp = temp.multiply(temp_db['i_post_scale'], axis = 0)
        # temp = temp.apply(get_frequency_content, frequency = 60, sample_rate = 7812.5, axis = 1).apply(pd.Series).iloc[:,1].apply(pd.Series)
        temp = db_to_waves(temp_db, e_or_i = 'i', hz60 = True, disturbance = True)
        temp = temp[1].dropna().apply(normalize, axis = 1).apply(pd.Series)
        #temp = waveform_chopper(temp, full = True, breaks = 10)
    except Exception as E:
        print('Error on sections {}-{} \n With error {}'.format(i, i+CHUNKSIZE, E))
        continue
    #ncol = temp.shape[1]
    #z = pd.DataFrame(np.zeros((temp.shape[0], WAVEFORM_LEN-ncol)))
    #i_data = pd.concat([temp, z], axis=1, ignore_index = True)
    
    autoencoder.fit(temp.dropna(), temp.dropna(), 
                    epochs = 10, batch_size = 100,
                    shuffle = True, 
                    validation_split = .1, verbose = 1)
    
    encode = Model(input_wave, encoded)
    
    encode_yaml = encode.to_yaml()
    
    with open("encode.yaml", "w") as yaml_file:
        yaml_file.write(encode_yaml)
    
    
    encode.save_weights("encode.h5")
    
    autoencoder_yaml = autoencoder.to_yaml()
    with open("autoencoder.yaml", "w") as yaml_file:
        yaml_file.write(autoencoder_yaml)
    
    autoencoder.save_weights("autoencoder.h5")
    print("Saved models to disk")    
    

