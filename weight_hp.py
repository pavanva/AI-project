from utils import *
import tf_utils

from os.path import join

from tensorflow.keras.metrics import BinaryAccuracy

from tensorflow_addons.metrics import CohenKappa
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Activation, add, GlobalAveragePooling2D, Dense, ZeroPadding2D, GaussianNoise
from tensorflow.keras.optimizers import Nadam
from itertools import islice

from res_gen import ResBlockGen

import math

from padding_generator import generator1
import pandas as pd

input_shape = (224, 224)
batch_size=64

def build_model(threshold):
    bn_momentum = 0.9
    initializer = 'he_uniform'

    block = ResBlockGen(bn_momentum, initializer).full_preactivation_res_block
    input_img = Input(shape=(*input_shape, 1), name='input')
    
    x = ZeroPadding2D(3, name='zero_padding_1')(input_img)
    x = Conv2D(filters=64, kernel_size=7, strides=2, name='conv_first', kernel_initializer=initializer)(x)
    x = BatchNormalization(name='bn_first', momentum=bn_momentum)(x)
    x = Activation('relu', name='activation_first')(x)

    x = ZeroPadding2D(1, name='zero_padding_2')(x)
    x = MaxPool2D(3, strides=2, name='max_pooling')(x)

    x = block(x, 64, 1, True)
    x = block(x, 64, 1)
    x = block(x, 64, 1)
    x = block(x, 128, 2)
    x = block(x, 128, 1)
    x = block(x, 128, 1)
    x = block(x, 128, 1)
    x = block(x, 256, 2)
    for i in range(5):
        x = block(x, 256, 1)
    x = block(x, 512, 2)
    x = block(x, 512, 1)
    x = block(x, 512, 1)

    x = Activation('relu', name='activation_last')(x)
    x = GlobalAveragePooling2D(name='GAP')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[input_img], outputs=[output])

    optimizer = Nadam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        metrics=[CohenKappa(num_classes=2), 'accuracy', BinaryAccuracy(threshold=threshold)],
        loss='binary_crossentropy',
    )
    
    return model

PATHS = data_paths('weight_search')

train = flow_from_csv(SetType.TRAIN, batch_size = 64, generator=generator1, input_shape=input_shape)    

#beta
basic_resource = 10

#n
denominator = 2

#freqency - how often model is evaluated
check_freq = 5

assert basic_resource % check_freq == 0

m = {}
#create models with threshold 0.5 to 0.9, step 0.05
for threshold in range(55, 91, 5):
    m[threshold/100] = (-1, build_model(threshold/100))

print(m)

global_best = -1

iteration = 0
#successive halving
while len(m) > 1:
    
    #evaluate models for desired numbers of epochs
    for threshold, ms in m.items():
        print('EVALUATING:', threshold)
        epochs_done = int(basic_resource * (denominator**iteration - 1))
        #Do validation every basic_resource steps
        #regular validation to decrease impact of noise
        for i in range((denominator**iteration) * check_freq):
            ms[1].fit(train, 
                 epochs=int(basic_resource / check_freq), 
                 steps_per_epoch=math.floor(len(train.labels) / batch_size), 
                 class_weight={0: threshold, 1: 1 - threshold},
                 initial_epoch=int(epochs_done),
                 verbose=0
            )

            #validation
            metrics = study_eval(
                ms[1], 
                SetType.VALID, 
                batch_size = 64, 
                aggregation = pd.core.groupby.generic.DataFrameGroupBy.max
            )

            print(threshold, metrics)

            study_kappa = metrics[3]
            #replace best score
            m[threshold] = (max(ms[0], study_kappa), ms[1])
            if global_best < study_kappa:
                global_best = study_kappa
                print('SAVING BEST:',study_kappa)
                ms[1].save(PATHS['best'])
            
        epochs_done += (basic_resource / check_freq)
    
    iteration += 1
    
    #remove worse part
    print('TRAILS:', m)
    select = int(len(m) / denominator)
    
    print('SELECTING:',select)
    #sort and select subset
    #https://stackoverflow.com/a/613218
    #https://stackoverflow.com/a/7971660
    sort = {k: v for k, v in sorted(m.items(), key=lambda item: item[1][0], reverse=True)}
    m = dict(islice(sort.items(), 0, select))
    
    print('REMAINING TRAILS:', m)

print('BEST PARAMS:', global_best)