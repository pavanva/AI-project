#This module processes gpu options related to tf
#Also adds custom object for keras serializtion

import os
import sys
import numpy as np
import subprocess as sp

ACCEPTABLE_AVAILABLE_MEMORY=2048

#Taken from https://github.com/yselivonchyk/TensorFlow_DCIGN
#forces usage of 'leave_unmasked' GPUs. Also ensures that selected GPUs are not fully alocated.
def mask_busy_gpus(leave_unmasked=1, random=True):
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = (sp.check_output(command.split())).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

        if len(available_gpus) < leave_unmasked:
            print('Found only %d usable GPUs in the system' % len(available_gpus))
            exit(0)

        if random:
            available_gpus = np.asarray(available_gpus)
            np.random.shuffle(available_gpus)

        # update CUDA variable
        gpus = available_gpus[:leave_unmasked]
        setting = ','.join(map(str, gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = setting
        print('Left next %d GPU(s) unmasked: [%s] (from %s available)'
        % (leave_unmasked, setting, str(available_gpus)))
    except FileNotFoundError as e:
        print('"nvidia-smi" is probably not installed. GPUs are not masked')
        print(e)
    except sp.CalledProcessError as e:
        print("Error on GPU masking:\n", e.output)

#hides all gpus
if '--cpu' in sys.argv:
    if '--memory' in sys.argv:
        raise Exception('--cpu and --memory can\'t be set together')
    if '--single-gpu' in sys.argv:
        raise Exception('--cpu and --single-gpu can\'t be set together')
        
    print('Forcing CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#forces usage of only one gpu
if '--single-gpu' in sys.argv:
    print('Forcing single GPU')
    mask_busy_gpus(1)
    
import tensorflow as tf

#restricts memory use
if '--memory' in sys.argv:
    mem_index = sys.argv.index('--memory')
    limit = int(sys.argv[mem_index + 1])
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
            print(e)

from tensorflow_addons.metrics import CohenKappa
from tensorflow.keras.utils import get_custom_objects       

#Addresses Cohen kappa serialization problem when loading model
kappa = CohenKappa(num_classes=2)
get_custom_objects().update({"CohenKappa": kappa})