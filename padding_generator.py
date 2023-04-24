import numpy as np
from PIL import Image
import io
import cv2

from skimage import exposure

from keras_preprocessing.image.utils import _PIL_INTERPOLATION_METHODS
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator

from keras_preprocessing.image.utils import img_to_array
from keras_preprocessing.image.affine_transformations import (apply_affine_transform,
                                     apply_brightness_shift,
                                     apply_channel_shift,
                                     flip_axis)

# original fun keras_preprocessing.image.load_img
# performs padding and preserves sample aspect ratio
def load_img(path, grayscale=False, color_mode='grayscale', target_size=None,
             interpolation='nearest'):
    """
        slightly modified load_img from keras_preprocessing.image.utils
        Changes the way of image resizing
    """
    with open(path, 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        if color_mode == 'grayscale':
            # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
            # convert it to an 8-bit grayscale image.
            if img.mode not in ('L', 'I;16', 'I'):
                img = img.convert('L')
        elif color_mode == 'rgba':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        elif color_mode == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                
                #resize image preserving image w/h ratio and filling with black
                w_ratio = width_height_tuple[0] / img.size[0]
                h_ratio = width_height_tuple[1] / img.size[1]
                min_ratio = min(w_ratio, h_ratio)
                img = img.resize((int(img.size[0] * min_ratio), int(img.size[1] * min_ratio)), resample)
                
                new_img = Image.new("L", width_height_tuple)
                new_img.paste(img, (int((width_height_tuple[0]-img.size[0])/2), int((width_height_tuple[1]-img.size[1])/2)))
                img = new_img
        return img

#same as original. Changed context calls local function load_img instead of keras_preprocessing.image.load_img
class PaddingDataFrameIterator(DataFrameIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,#(512, 512),
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

#Overriden to perform custom augmentations
class PaddingImageDataGenerator(ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 interpolation_order=1,
                 equalize=False,
                 gaussian_noise=False,
                 gaussian_noise_var=0.1,
                 adp_equalize=False,
                 dtype='float32'):
        self.equalize=equalize
        self.adp_equalize=adp_equalize
        
        assert not (equalize and adp_equalize)
        
        self.gaussian_noise=gaussian_noise
        self.gaussian_noise_var=gaussian_noise_var
        super(PaddingImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            interpolation_order=interpolation_order,
            dtype=dtype)
    
    #modified original function
    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.
        # Arguments
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intensity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.
        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                                   transform_parameters.get('tx', 0),
                                   transform_parameters.get('ty', 0),
                                   transform_parameters.get('shear', 0),
                                   transform_parameters.get('zx', 1),
                                   transform_parameters.get('zy', 1),
                                   row_axis=img_row_axis,
                                   col_axis=img_col_axis,
                                   channel_axis=img_channel_axis,
                                   fill_mode=self.fill_mode,
                                   cval=self.cval,
                                   order=self.interpolation_order)
        
        if self.equalize:
            x = x / 255
            x = exposure.equalize_hist(x)
            x = x * 255
            
        if self.adp_equalize:
            x = x / 255
            x = exposure.equalize_adapthist(x, kernel_size=35)
            x = x * 255
            
        if transform_parameters.get('channel_shift_intensity') is not None:
            x = apply_channel_shift(x,
                                    transform_parameters['channel_shift_intensity'],
                                    img_channel_axis)

        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, img_row_axis)

        if transform_parameters.get('brightness') is not None:
            x = apply_brightness_shift(x, transform_parameters['brightness'])

        if self.gaussian_noise:
            #Taken from https://stackoverflow.com/a/30609854
            mean = 0
            var = self.gaussian_noise_var * 255
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,x.shape)
            gauss = gauss.reshape(*x.shape)
            x = x + gauss
        
        return x

    #return custom class
    def flow_from_dataframe(self,
                            dataframe,
                            directory=None,
                            x_col="filename",
                            y_col="class",
                            weight_col=None,
                            target_size=(224, 224),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest',
                            validate_filenames=True,
                            **kwargs):
        return PaddingDataFrameIterator(
            dataframe,
            directory,
            self,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames,
            **kwargs
        )
#
# Definitions of used augmentations
#
    
#Only preprocessing without augmentation
basic_generator = PaddingImageDataGenerator(
    rescale=1./255,
    fill_mode='constant',
    cval=0
)

basicg_generator = PaddingImageDataGenerator(
    rescale=1./255,
    gaussian_noise=True,
    gaussian_noise_var=0.06,
    fill_mode='constant',
    cval=0
)

basice_generator = PaddingImageDataGenerator(
    rescale=1./255,
    fill_mode='constant',
    equalize=True,
    cval=0
)

generator1 = PaddingImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    fill_mode='constant',
    cval=0
)

generator1g = PaddingImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    gaussian_noise=True,
    gaussian_noise_var=0.06,
    fill_mode='constant',
    cval=0
)

generator1e = PaddingImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    fill_mode='constant',
    equalize=True,
    cval=0
)
    
generator2 = PaddingImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=(0.7, 1.1),
    shear_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.7, 1.2),
    fill_mode='constant',
    cval=0
)
    
generator2e = PaddingImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=(0.7, 1.1),
    shear_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    equalize=True,
    fill_mode='constant',
    cval=0
)
    
generator2g = PaddingImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=(0.7, 1.1),
    shear_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.7, 1.2),
    gaussian_noise=True,
    gaussian_noise_var=0.06,
    fill_mode='constant',
    cval=0
)

# Generator codes: 
# b - no augmentation
# 1 - Modest aug
# 2 - Strong aug
# 
# suffix e for equalization, eg: 1e
# suffix g for gauss noise, eg: 2g

#dict of generators for easy access
generators = {
    '1': generator1,
    '2': generator2,
    'b': basic_generator,
    '1g': generator1g,
    '2g': generator2g,
    'bg': basicg_generator,
    '1e': generator1e,
    '2e': generator2e,
    'be': basice_generator,
}