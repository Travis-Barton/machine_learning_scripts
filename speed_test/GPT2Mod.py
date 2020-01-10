#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:54:34 2020

@author: tbarton
"""


import gpt_2_simple as gpt2
#gpt2.download_gpt2()


sess = gpt2.start_tf_sess()
gpt2.finetune(sess, 'TheBible.txt', steps=1000)   # steps is max number of training steps

single_text = gpt2.generate(sess, return_as_list=True)[0]
print(single_text)