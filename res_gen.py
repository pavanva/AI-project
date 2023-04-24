from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, add

class ResBlockGen():
    #class for res blocks generation
    def __init__(
        self,
        bn_momentum = 0.1,
        initializer = 'he_uniform'
    ):
        self.bn_momentum = bn_momentum
        self.initializer = initializer
        self.block_number = 0
        
    def reset_block_number(self):
        self.block_number = 0
    
    #original block
    def res_block_2l(self, input_block, filters, strides, force_short_conv = False):
        self.block_number += 1
        block_name = 'block' + str(self.block_number) + '_'

        y = Conv2D(
            filters=filters, 
            kernel_size=3,
            strides=strides, 
            padding='same', 
            name=block_name + 'conv_1',
            kernel_initializer=self.initializer
        )(input_block)
        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_1')(y)
        y = Activation('relu', name=block_name + 'activation_1')(y)

        y = Conv2D(
            filters=filters, 
            kernel_size=3, 
            padding='same',
            name=block_name + 'conv_2',
            kernel_initializer=self.initializer
        )(y)
        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_2')(y)

        if strides != 1 or force_short_conv:
            z = Conv2D(kernel_size=1, filters=filters, strides=strides, name=block_name + 'conv_S',
                kernel_initializer=self.initializer)(input_block)
            z = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_S')(z)
        else:
            z = input_block
        x = add([z, y], name=block_name + 'add')
        return Activation('relu', name=block_name + 'activation_after')(x)

    
    #original bottleneck block
    def res_block_3l(self, input_block, filters, strides, force_short_conv = False):
        self.block_number += 1
        block_name = 'block' + str(self.block_number) + '_'

        y = Conv2D(
            filters=filters, 
            kernel_size=1,
            strides=strides, 
            padding='same', 
            name=block_name + 'conv_1',
            kernel_initializer=self.initializer
        )(input_block)
        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_1')(y)
        y = Activation('relu', name=block_name + 'activation_1')(y)

        y = Conv2D(
            filters=filters, 
            kernel_size=3, 
            padding='same',
            name=block_name + 'conv_2',
            kernel_initializer=self.initializer
        )(y)
        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_2')(y)
        y = Activation('relu', name=block_name + 'activation_2')(y)

        y = Conv2D(
            filters=filters*4, 
            kernel_size=1,
            strides=strides, 
            padding='same', 
            name=block_name + 'conv_3',
            kernel_initializer=self.initializer
        )(y)
        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_3')(y)

        if strides != 1 or force_short_conv:
            z = Conv2D(kernel_size=1, filters=filters*4, strides=strides, name=block_name + 'conv_S',
                kernel_initializer=self.initializer)(input_block)
            z = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_S')(z)
        else:
            z = input_block
        x = add([z, y], name=block_name + 'add')
        return Activation('relu', name=block_name + 'activation_after')(x)

    #full preactivation bottleneck block
    def full_preactivation_res_block_3l(self, input_block, filters, strides, force_short_conv = False):
        self.block_number += 1
        block_name = 'block' + str(self.block_number) + '_'

        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_1')(input_block)
        y = Activation('relu', name=block_name + 'activation_1')(y)
        y = Conv2D(
            filters=filters, 
            kernel_size=1,
            strides=strides, 
            padding='same', 
            name=block_name + 'conv_1',
            kernel_initializer=self.initializer
        )(y)

        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_2')(y)
        y = Activation('relu', name=block_name + 'activation_2')(y)
        y = Conv2D(
            filters=filters, 
            kernel_size=3, 
            padding='same',
            name=block_name + 'conv_2',
            kernel_initializer=self.initializer
        )(y)

        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_3')(y)
        y = Activation('relu', name=block_name + 'activation_3')(y)
        y = Conv2D(
            filters=filters*4, 
            kernel_size=1,
            padding='same', 
            name=block_name + 'conv_3',
            kernel_initializer=self.initializer
        )(y)

        if strides != 1 or force_short_conv:
            z = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_S')(input_block)
            z = Activation('relu', name=block_name + 'activation_S')(z)
            z = Conv2D(kernel_size=1, filters=filters*4, strides=strides, name=block_name + 'conv_S',
                kernel_initializer=self.initializer)(z)
        else:
            z = input_block
        return add([z, y], name=block_name + 'add')

    #full preactivation block
    def full_preactivation_res_block(self, input_block, filters, strides, shortcut_conv=False):
        self.block_number += 1
        block_name = 'block' + str(self.block_number) + '_'

        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_1')(input_block)
        split = Activation('relu', name=block_name + 'activation_1')(y)
        y = Conv2D(
            filters=filters, 
            kernel_size=3,
            strides=strides, 
            padding='same', 
            name=block_name + 'conv_1',
            kernel_initializer=self.initializer
        )(split)

        y = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_2')(y)
        y = Activation('relu', name=block_name + 'activation_2')(y)
        y = Conv2D(
            filters=filters, 
            kernel_size=3, 
            padding='same', 
            name=block_name + 'conv_2',
            kernel_initializer=self.initializer
        )(y)

        if strides != 1:
            z = BatchNormalization(momentum=self.bn_momentum, name=block_name + 'bn_S')(split)
            z = Activation('relu', name=block_name + 'activation_S')(z)
            z = Conv2D(kernel_size=1, filters=filters, strides=strides, name=block_name + 'conv_S',
                kernel_initializer=self.initializer)(split)
        else:
            z = input_block
        return add([z, y], name=block_name + 'add')