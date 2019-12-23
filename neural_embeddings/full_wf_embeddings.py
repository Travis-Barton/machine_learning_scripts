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
from Autoencoder_to_TSNE_Reduction import wf_autoencoder
import pandas as pd
import keras
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
MAX_VAL = 90000
CHUNKSIZE = 1000
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
from pq_utils import check_lol, get_windowed_rms_waveform_centered, get_frequency_content, get_total_frequency_content
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
    """
    sqlr = sql + """where dwr.id >= {start_id} and dwr.id < {end_id} """.format(start_id=start_id, end_id=end_id)
    return pd.read_sql(sqlr, wfl_conn)





input_wave = Input(shape=(28000,))   
encoded1 = Dense(9000, activation = 'sigmoid')(input_wave)  
encoded2 = Dense(9000, activation = 'relu')(encoded1)  
encoded3 = Dense(7000, activation = 'relu', activity_regularizer=keras.regularizers.l2(0.01))(encoded2)   
encoded4 = Dense(5000, activation = 'relu', activity_regularizer=keras.regularizers.l1(0.01))(encoded3)  
encoded5 = Dense(3000, activation = 'relu', )(encoded4)  
encoded = Dense(EMBEDDING_DIM, activation = 'relu')(encoded5)  
decoded2 = Dense(3000, activation = 'relu')(encoded)   
decoded3 = Dense(5000, activation = 'relu')(decoded2)  
decoded4 = Dense(7000, activation = 'relu')(decoded3)   
decoded5 = Dense(9000, activation = 'relu')(decoded4)  
decoded6 = Dense(9000, activation = 'relu')(decoded5)  
decoded = Dense(12000, activation = 'sigmoid')(decoded6)
  
autoencoder = Model(input_wave, decoded)
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')



for i in np.arange(0, MAX_VAL, CHUNKSIZE):    
    print('trying {}-{}'.format(i, i+CHUNKSIZE))
    temp_db = get_dist_wfs_by_dwr_table_id(dbc.db_connect(dbc.wfl_db_config), i, i+CHUNKSIZE)
    # temp = pd.DataFrame(temp_db['e_wf_raw_bytes'].apply(np.frombuffer, dtype = '<i4').apply(pd.Series))
    # temp = temp.multiply(temp_db['e_post_scale'], axis = 0)
    # e_data = e_data.append(temp)  
    
    try:
        temp = pd.DataFrame(temp_db['i_wf_raw_bytes'].apply(np.frombuffer, dtype = '<i4').apply(pd.Series))
        temp = temp.multiply(temp_db['i_post_scale'], axis = 0)
        
    except:
        print('Error on sections {}-{}'.format(i, i+CHUNKSIZE))
        continue
    ncol = temp.shape[1]
    z = pd.DataFrame(np.zeros((temp.shape[0], WAVEFORM_LEN-ncol)))
    i_data = pd.concat([temp, z], axis=1, ignore_index = True)
    autoencoder.fit(i_data, i_data, 
                    epochs = 50, batch_size = 600,
                    shuffle = True, 
                    validation_split = .1, verbose = 0)
    encode = Model(input_wave, encoded)
    
    encode_yaml = encode.to_yaml()
    
    with open("encode_yaml.yaml", "w") as yaml_file:
        yaml_file.write(encode_yaml)
    
    
    encode.save_weights("encode.h5")
    
    autoencoder_yaml = autoencoder.to_yaml()
    with open("autoencoder_yaml.yaml", "w") as yaml_file:
        yaml_file.write(autoencoder_yaml)
    
    autoencoder.save_weights("autoencoder.h5")
    print("Saved models to disk")    
    

