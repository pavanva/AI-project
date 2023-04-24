import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow import keras
from PIL import Image
from tensorflow.keras.layers import Conv2D, add

def show_image(img):
    img = Image.fromarray(img)
    img.show()

def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def image_shape(model):
    shape = model.layers[0].input_shape[0]
    print(shape)
    return (shape[1], shape[2])

#Taken from (modified)
#captures CAM using Grad-CAM
#
#https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
def explain(img, model: Model, label, cmap=cv2.COLORMAP_HOT, concat_original=False, layer_name=None, reduction = 0.0):
    for layer in model.layers:
        if type(layer) is Conv2D:
            target_layer = layer
    
    model = Model(
        inputs=[model.inputs],
        outputs=[target_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        inputs = tf.cast(img, tf.float32)
        (convOutput, prediction) = model(inputs)
        loss = prediction[0]
    grad = tape.gradient(loss, convOutput)
    
    castConvOutput = tf.cast(convOutput > 0, "float32")
    castGrad = tf.cast(grad > 0, "float32")
    guidedGrad = castConvOutput * castGrad * grad
    
    convOutput = convOutput[0]
    guidedGrad = guidedGrad[0]
    
    weights = tf.reduce_mean(guidedGrad, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutput), axis=-1)
    
    heatmap = cv2.resize(cam.numpy(), image_shape(model))
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + 1e-8
    heatmap = numer / denom * prediction[0][0].numpy()
    
    col_img = cv2.cvtColor(np.uint8(np.squeeze(img)*255),cv2.COLOR_GRAY2RGB)
    
    col_heat = cv2.applyColorMap(np.uint8(heatmap*255), cmap)
    
    #reduce noise
    m = heatmap.max()
    heatmap = np.maximum(heatmap - reduction, 0) / (m - reduction) * m
    
    reduced = cv2.applyColorMap(np.uint8(heatmap*255), cmap)
    
    mapped = cv2.addWeighted(col_img, 1, reduced, 0.35, 1)
    
    if concat_original:
        mapped = cv2.vconcat([mapped, col_img])
    
    return (mapped, heatmap, prediction[0][0].numpy())