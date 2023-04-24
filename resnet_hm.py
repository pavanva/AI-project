from kerastuner import HyperModel

from tensorflow_addons.metrics import CohenKappa

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Activation, add, GlobalAveragePooling2D, Dense, ZeroPadding2D

from res_gen import ResBlockGen

from tensorflow.keras.optimizers import Nadam

#HyperModel for HP search

class ResnetHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        initializer = 'he_uniform'
        
        block = ResBlockGen(hp.get('bn_momentum'), initializer).full_preactivation_res_block

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
        
        optimizer = Nadam(
            learning_rate=hp.get('opt_learning_rate'),
            beta_1=hp.get('opt_beta_1'),
            beta_2=hp.get('opt_beta_2'),
            epsilon=hp.get('opt_epsilon')
        )

        model.compile(
            optimizer=optimizer,
            metrics=[CohenKappa(num_classes=2), 'accuracy'],
            loss='binary_crossentropy'
        )

        return model