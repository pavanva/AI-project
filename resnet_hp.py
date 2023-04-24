from utils import *

from resnet_hm import ResnetHyperModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from os.path import join

import kerastuner
from kerastuner import HyperParameters

import math

from padding_generator import generator1

input_shape = (224, 224)

train = flow_from_csv(SetType.TRAIN, generator=generator1, input_shape=input_shape)
valid = flow_from_csv(SetType.VALID, input_shape=input_shape)

#custom tuner to customize batch size and provide possibility to save b
class _MyTuner(kerastuner.tuners.RandomSearch):
    
    def run_trial(self, trial, *args, **kwargs):
        batch_size = trial.hyperparameters.get('batch_size')
        
        #sets batch size for generators
        
        train.batch_size = batch_size
        valid.batch_size = batch_size
            
        kwargs['x'] = train
        kwargs['validation_data'] = valid
        kwargs['steps_per_epoch'] = math.floor(len(train.labels) / batch_size)
        kwargs['validation_steps'] = math.floor(len(valid.labels) / batch_size)

        #saves trail config
        with open(os.path.join(self.directory, self.project_name, 'trial_' + trial.trial_id + '_config.json'), 'w+') as f:
            str_config = str(trial.hyperparameters.get_config())
            f.write(str_config)
        
        path = os.path.join(self.directory, self.project_name, 'trial_' + trial.trial_id + '_log_{}.csv')
        
        #finds unused name for log file
        i = 0
        while True:
            i+=1
            path_form = path.format(i)
            if not os.path.exists(path_form):
                break

        #passes logger callback
        kwargs['callbacks'].append(CSVLogger(path_form))
        
        #passes run_trial logic to super class
        super(_MyTuner, self).run_trial(trial, *args, **kwargs)
        
        #removes the callback
        kwargs['callbacks'].pop()

#select HP configs
        
hp = HyperParameters()

hp.Choice('opt_learning_rate', [1/10**i for i in range(1, 5)], default=1e-3) 
hp.Choice('batch_size', [2**i for i in range(5, 9)])
hp.Fixed('opt_beta_1', 0.9)
hp.Fixed('opt_beta_2', 0.999)
hp.Fixed('opt_epsilon', 1e-8)
hp.Choice('bn_momentum', [0.01, 0.1, 0.9]) #initially I was tuning batch normalization momentum also, since I suspected it to create distubtions in training, however it was not proved to be true

#random seach with more combinations then the HS has -> grid search
#3 trials to increase evaluation robusteness 
tuner = _MyTuner(
    ResnetHyperModel(input_shape),
    objective=kerastuner.Objective('val_cohen_kappa', direction='max'),
    executions_per_trial=3,
    max_trials=100,
    directory=MODEL_FOLDER,
    project_name='resnet_hp',
    hyperparameters=hp,
    allow_new_entries=False
)

tuner.search(
    callbacks=[EarlyStopping(monitor='val_cohen_kappa', mode='max', patience=10, min_delta=0.02)], 
    epochs=50
)
