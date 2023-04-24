from utils import *
import tf_utils

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow import keras
from tensorflow.keras import backend as K

import tensorflow as tf

from padding_generator import generators

import math

import sys
import os

PATHS = data_paths(sys.argv[1], create_if_missing=False)

#LR_OVERRIDE = float(get_arg('--const-lr', 0))

#ratio to multiply lr, with every interval (epochs)
DECAY = get_arg('--decay', [0, 0], count=2)
DECAY_RATE = float(DECAY[0])
DECAY_INTERVAL = int(DECAY[1])
if DECAY_INTERVAL != 0:
    print('DECAY_RATE: {}, DECAY_INTERVAL: {}'.format(DECAY_RATE, DECAY_INTERVAL))
    
SCHEDULE_PATIENCE = 13
SCHEDULE_DELTA = 0.01
LRS = '--lrs' in sys.argv
if LRS:
    print('Using LR schedule')

METRIC = 'val_cohen_kappa'

#lrs which pass the learning rate
class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    #Taken from https://www.tensorflow.org/guide/keras/custom_callback#learning_rate_scheduling
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('Learning rate is %6.4f.' % (scheduled_lr))
        
since_step_down = 0
def schedule(epoch, lr):
    # decay
    if DECAY_INTERVAL != 0 and epoch % DECAY_INTERVAL == 0 and epoch != 0:
        lr = lr * DECAY_RATE
    
    # adaptive lrs
    if LRS:
        global since_step_down
        since_step_down += 1

        if since_step_down < SCHEDULE_PATIENCE:
            return lr
        log = get_log(PATHS)
        before_max = log.iloc[:-SCHEDULE_PATIENCE][METRIC].max()
        best = log[METRIC].max()

        if before_max + SCHEDULE_DELTA > best:
            since_step_down = 0
            return lr * 0.1
    return lr
    
def train():
    #load last model or --init if specified
    model_path = get_arg('--init', PATHS['last'] if os.path.exists(PATHS['last']) else PATHS['init'])
    if not os.path.isfile(model_path):
        print('File ' + model_path + ' does not exist. Assuming it is a project name...')
        model_path = data_paths(model_path, create_if_missing=False)['init']
        
    print('Loading model ' + model_path)
    model = load_model(model_path)
    
    #append training args to notes
    set_note(PATHS, str(sys.argv), 'a')

    gen_code = get_arg('--gen', 'b')
    print('TRAIN GENERATOR: ' + gen_code) 
    gen = generators[gen_code]

    val_gen_code = get_arg('--vgen', 'b')
    print('VALID GENERATOR: ' + val_gen_code) 
    val_gen = generators[val_gen_code]
    
    epochs = int(get_arg('--epochs', 100))
    
    batch_size = int(get_arg('--batch-size', 64))
    print('BATCH SIZE: ' + str(batch_size))
    
    apply_sample_weights = '--sample-weights' in sys.argv
    print('USING SAMPLE WEIGHTS: ' + str(class_weights))
    
    apply_class_weights = '--class-weights' in sys.argv

    train_generator = flow_from_csv(SetType.TRAIN, generator=gen, batch_size=batch_size, input_shape=model, class_weights=apply_class_weights, sample_weights=apply_sample_weights)
    valid_generator = flow_from_csv(SetType.VALID, generator=val_gen, batch_size=batch_size, input_shape=model)

    #expand tuple
    # flow_from_csv returns tuple if class_weights
    if apply_class_weights:
        class_weights = train_generator[1][1]
        train_generator = train_generator[0]
        print('CLASS WEIGHTS: ' + str(class_weights))
    
    #early stopping
    patience = int(get_arg('--es-patience', 30))
    min_delta = int(get_arg('--es-min_delta', 0.01))
    es = EarlyStopping(monitor=METRIC, mode='max', verbose=1, patience=patience, min_delta=min_delta),
    
    best_checkpoint = ModelCheckpoint(PATHS['best'], monitor=METRIC, mode='max', verbose=1, save_best_only=True)
    #load best score
    best_checkpoint.best = best_score(PATHS['log'])
    
    callbacks = [
        CSVLogger(PATHS['log'], append=True),
        best_checkpoint,
        ModelCheckpoint(PATHS['last'], verbose=0),
        LearningRateScheduler(schedule),
        es
    ]

    hist = model.fit(train_generator, 
                 validation_data=valid_generator, 
                 callbacks=callbacks, 
                 epochs=epochs, 
                 steps_per_epoch=math.floor(len(train_generator.labels) / batch_size), 
                 validation_steps=math.floor(len(valid_generator.labels) / batch_size),
                 class_weight=None if not apply_class_weights else class_weights,
                 initial_epoch=last_epoch(PATHS) + 1
                )

#mirror to use multiple gpus
if '--mirror' in sys.argv:
    print('Using tf mirror strategy')
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        train()
else:
    train()