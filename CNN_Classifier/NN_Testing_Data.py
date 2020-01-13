#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:53:45 2019

@author: tbarton
"""

import pandas as pd
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
from multiprocessing import  Pool


sql_labels = '''
SELECT COUNT(label), label FROM wfl_v3.disturbance_labels
GROUP BY label;
'''

# sql_data = '''
# SELECT e_wf_raw_bytes, e_post_scale, e_post_offset, i_wf_raw_bytes, i_post_scale, i_post_offset, label FROM wfl_v3.disturbance_labels m
# LEFT JOIN disturbance_mm3_basics db on db.disturbance_id =  m.disturbance_id
# LEFT JOIN disturbance_waveforms_raw r on r.waveform_id = m.waveform_id
# where i_len < 10021 and e_len < 10021 and label = '{}'
# '''.format(i)


db = pd.read_sql(sql_labels, dbc.db_connect(dbc.wfl_db_config))

db = db.loc[db['COUNT(label)'] > 10000]


e_data = pd.DataFrame()
i_data = pd.DataFrame()
for i in tqdm(db['label'], position=0, leave=True):
    sql_data = '''
    SELECT e_wf_raw_bytes, e_post_scale, e_post_offset, i_wf_raw_bytes, i_post_scale, i_post_offset, label FROM wfl_v3.disturbance_labels m
    LEFT JOIN disturbance_mm3_basics db on db.disturbance_id =  m.disturbance_id
    LEFT JOIN disturbance_waveforms_raw r on r.waveform_id = m.waveform_id
    where i_len < 10021 and e_len < 10021 and label = '{}'
    LIMIT 10000
    '''.format(i)
    temp_db = pd.read_sql(sql_data, dbc.db_connect(dbc.wfl_db_config))
    temp = pd.DataFrame(temp_db['e_wf_raw_bytes'].apply(np.frombuffer, dtype = '<i4').apply(pd.Series))
    temp = temp.multiply(temp_db['e_post_scale'], axis = 0)
    temp['label'] = i
    e_data = e_data.append(temp)  
    
    temp = pd.DataFrame(temp_db['i_wf_raw_bytes'].apply(np.frombuffer, dtype = '<i4').apply(pd.Series))
    temp = temp.multiply(temp_db['i_post_scale'], axis = 0)
    temp['label'] = i
    i_data = i_data.append(temp)
    
    