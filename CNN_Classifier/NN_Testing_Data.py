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

labels = db['label']



sql = '''

SELECT m.disturbance_id, e_wf_raw_bytes, e_post_scale, e_post_offset, i_wf_raw_bytes, i_post_scale, i_post_offset, label FROM wfl_v3.disturbance_labels m
    JOIN disturbance_mm3_basics db on db.disturbance_id =  m.disturbance_id
    JOIN disturbance_waveforms_raw r on r.waveform_id = m.waveform_id
    where i_len < 10021 and e_len < 10021 and label in ('9th harm', 'bad channel switching', 'broken', 'fault', 'inrush', 'load change')
    LIMIT 10000
'''
db = pd.read_sql(sql, dbc.db_connect(dbc.wfl_db_config))

e_data = db_to_waves(db, 'e', True, disturbance = True)
e_data[1] = pd.DataFrame(normalize(e_data[1])).fillna(0)


i_data = db_to_waves(db, 'i', True, disturbance = True)
i_data[1] = pd.DataFrame(normalize(i_data[1])).fillna(0)


codes_done()