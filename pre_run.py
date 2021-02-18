#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:09:35 2021

@author: Simone Rossetti
"""
import os
import config as cfg

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPU

import gc 
gc.collect()

import tensorflow as tf

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#	try:
#		# Currently, memory growth needs to be the same across GPUs
#		for gpu in gpus:
#			tf.config.experimental.set_memory_growth(gpu, True)
#		logical_devices = tf.config.experimental.list_logical_devices('GPU')
#		print(len(gpus), "Physical GPUs,", len(logical_devices), "Logical GPUs")
#	except RuntimeError as e:
#		# Memory growth must be set before GPUs have been initialized
#		print(e)
#else: 
#    print('No GPU found')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 3 virtual GPUs with 1GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)