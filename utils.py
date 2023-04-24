import os
import sys

def get_arg(arg, default=None, count=1):
    """
    returns value of argument or default (list if count > 1)
    """
    if arg in sys.argv:
        index = sys.argv.index(arg)
        args = sys.argv[index + 1: index + 1 + count]
        return args if count > 1 else args[0]
    else:
        return default

MODEL_FOLDER = get_arg('--dataset-folder', os.path.join('..',  'models'))
DATA_FOLDER = get_arg('--model-folder', os.path.join('..', 'MURA-v1.1'))

from enum import Enum
import glob
import numpy as np
import pandas as pd
from PIL import Image
import re
import string
import io
from os.path import sep
from padding_generator import basic_generator
from tensorflow_addons.metrics import CohenKappa
from tensorflow.keras.utils import get_custom_objects
import math

class XRType(Enum):
    WRIST = 'XR_WRIST'
    SHOULDER = 'XR_SHOULDER'
    HUMERUS = 'XR_HUMERUS'
    HAND = 'XR_HAND'
    FOREARM = 'XR_FOREARM'
    FINGER = 'XR_FINGER'
    ELBOW = 'XR_ELBOW'
    ALL = 'XR_*'

#test split logic is defined in load_data_from_csv    
class SetType(Enum):
    TRAIN = {'path': 'train', 'id': 1}
    VALID = {'path': 'valid', 'id': 2}
    TEST = {'path': 'train', 'id': 3} 

    
def data_paths(model_name: str, create_if_missing = True, model_folder = MODEL_FOLDER):
    """
    Returns map of paths connected to project to 'best', 'last', 'log', 'init', 'note' based on MODEL_FORLDER
    
    Defines an enviroment for a model
    """
    folder = os.path.join(model_folder, 'model_' + model_name)
    
    if not os.path.isdir(folder):
        if create_if_missing:
            os.mkdir(folder)
        else:
            raise Exception('folder ' + folder + ' does not exists')
    
    return {
        'dir': folder,
        'best': os.path.join(folder, 'best.h5'),
        'last': os.path.join(folder, 'last.h5'),
        'log': os.path.join(folder, 'history.log'),
        'init': os.path.join(folder, 'init.h5'),
        'note': os.path.join(folder, 'note.txt')
    }

common_flow_kwargs = {
    'x_col': "image_path",
    'y_col': "label",
    'classes': ['True', 'False'],
    'color_mode': "grayscale",
    'class_mode': 'binary'
}

def flow_from_csv(set_type: SetType,
                  xr_type: XRType = XRType.ALL,
                  generator = basic_generator,
                  data_folder = DATA_FOLDER,
                  shuffle=True,
                  input_shape=(224, 224),
                  batch_size=32,
                  sample_weights=False,
                  class_weights=False
                 ):
    """
    Input shape might be a tuple or a model. If it is a model the image shape is derived from input layer
     = XRType.ALL
    Returns ImageDataIterator based on data in DATA_FORLDER/{xr_type}/{set_type}_image_paths.csv
    """
    #infer shape
    if not type(input_shape) is tuple:
        input_shape = model_image_shape(input_shape)
    
    data = load_data_from_csv(set_type, xr_type, data_folder, sample_weights)
    
    flow = generator.flow_from_dataframe(
        dataframe=data,
        weight_col='weight' if sample_weights else None,
        target_size=input_shape,
        batch_size=batch_size,
        shuffle=shuffle,
        **common_flow_kwargs
    )
    if class_weights:
        return (flow, get_class_weights(data))
    return flow

def load_data_from_csv(set_type: SetType, xr_type: XRType = XRType.ALL, data_folder = DATA_FOLDER, sample_weights=False):
    """
    returns pd.DataFrame with columns 'study_id', 'image_num', 'label', 'image_path'
        one image per row
        from DATA_FORLDER/{set_type}_image_paths.csv
        sample_weighs adds column weight
        returns df or tupple (df, class_weights) if class_weights
    """
    df_images = pd.read_csv(os.path.join(data_folder, set_type.value['path'] + '_image_paths.csv'), header=None)
    
    #process individual lines to get 'study_id', 'xr_type', 'image_num', 'label', 'image_path'
    image_list = list()
    for image_path in df_images[df_images[0].str.contains(xr_type.value)][0]:
        path_segments = image_path.split(sep)
        label = 'positive' in path_segments[-2]
        patient_id = re.findall(r'\d+', path_segments[-3])
        xr = path_segments[-4]
        study_number = re.findall(r'\d+', path_segments[-2])
        study_id = patient_id[0] + xr + study_number[0]
        image_list.append((study_id, xr, int(re.findall(r'\d+', path_segments[-1])[0]), str(label), data_folder + sep + sep.join(path_segments[1:])))
    
    df =  pd.DataFrame.from_records(image_list, columns = [ 'study_id', 'xr_type', 'image_num', 'label', 'image_path' ] )
    
    # takes (for test) or excludes (for train) images defined in DATA_FOLDER/test_studies.csv
    if set_type != SetType.VALID:
        t = pd.read_csv(os.path.join(DATA_FOLDER, 'test_studies.csv'), header=None)
        vs = t.values[:, 0]
        filt = df['study_id'].apply(lambda x: x in vs)
        df = df[~filt] if set_type == SetType.TRAIN else df[filt]
    
    if sample_weights:
        df['weight'] = df.groupby(['study_id'])["image_path"].transform("count").astype('float32')**-1
    return df

def predict_all(model, 
            set_type: SetType, 
            xr_type: XRType = XRType.ALL, 
            data_folder = DATA_FOLDER,
            generator = basic_generator,
            batch_size=32):
    """
    Keras does not evaluate samples indivisible with batch_size
    This method will evaluate all samples
    """
    
    image_shape = model_image_shape(model)
    
    data = load_data_from_csv(set_type, xr_type, data_folder)
    image_count = len(data.index)
    
    flow = generator.flow_from_dataframe(
        dataframe=data,
        target_size=image_shape,
        batch_size=batch_size,
        shuffle=False,
        **common_flow_kwargs
    )
    
    #Keras does not evaluate samples indivisible with batch_size
    rest = image_count % batch_size
    flow_rest = generator.flow_from_dataframe(
        dataframe=data[-rest:],
        target_size=image_shape,
        batch_size=rest,
        shuffle=False,
        **common_flow_kwargs
    )
    
    #model predict does not work without deffined steps
    pred = model.predict(flow, steps=math.floor(len(flow.labels) / batch_size))
    #predicts remaining samples
    if rest != 0:
        pred2 = model.predict(flow_rest, steps = 1)
        
        #All predictions
        p = np.concatenate([pred, pred2])
    else:
        p = pred

    assert len(p) == image_count
    
    return p

def study_eval(model, 
            set_type: SetType, 
            xr_type: XRType = XRType.ALL, 
            data_folder = DATA_FOLDER,
            generator = basic_generator,
            batch_size=32, 
            aggregation = pd.core.groupby.generic.DataFrameGroupBy.mean,):
    """
    Evaluates model for studies instead of individual images
    
    returns (individual_accuracy, individual_cohen_kappa, study_accuracy, study_cohen_kappa)
    """
    #process data
    path = data_folder + sep + set_type.value['path'] + '_labeled_studies.csv'
    data = load_data_from_csv(set_type, xr_type, data_folder)
    
    p = predict_all(model, set_type, xr_type, data_folder, generator, batch_size)
    
    studies = pd.read_csv(path, header=None, names=['study', 'label'])
    Y = data['label'].to_numpy()
    
    data['pred'] = p
    
    data['label_int'] = data['label'].apply(lambda x: 1 if x == 'True' else 0).astype(np.int32)
    
    data['pred_int'] = data['pred'].apply(lambda x: round(x))
    
    count = len(data.index)
    t = len(data[data['pred_int'] == data['label_int']].index)
    i_pr_a = t/count
    
    #calculate kappa and accuracy for indiv images
    i_kappa = cohens_kappa(data, 'label_int', 'pred_int')
    
    data = aggregation(data.groupby(by=['study_id']))
    data['pred_final'] = data['pred'].apply(lambda x: round(x))

    #calculate kappa and accuracy for studies
    studies = len(data.index)
    t = len(data[data['pred_final'] == data['label_int']].index)
    pr_a = t/studies
    
    kappa = cohens_kappa(data, 'label_int', 'pred_final')
        
    return (i_pr_a, i_kappa, pr_a, kappa)

def cohens_kappa(df, label_col, pred_col):
    """
    Calculates kappa from df['label_col'] and df['pred_col']
    """
    count = len(df.index)
    t = len(df[df[pred_col] == df[label_col]].index)
    pr_a = t/count
    p_p = (len(df[df[pred_col] == 1].index)/count) * (len(df[df[label_col] == 1].index)/count)
    p_n = (len(df[df[pred_col] == 0].index)/count) * (len(df[df[label_col] == 0].index)/count)
    pr_e = p_p + p_n
    kappa = (pr_a - pr_e) / (1 - pr_e)

    return kappa

def available_models(keywords = []):
    """
    Finds models in MODEL_FOLDER which have best
    
    keywords defines a filter (result contains all keywords)
    """
    available_paths = glob.glob(os.path.join(MODEL_FOLDER, 'model_*','best.h5'))
    available_ms = [x[len(MODEL_FOLDER + os.path.sep + 'model_') : -len(os.path.sep + 'best.h5')] for x in available_paths]
    available_ms = pd.DataFrame(available_ms, columns=['name'])
    available_ms['best_score'] = available_ms['name'].apply(lambda x: best_score(data_paths(x)))

    available_ms = available_ms.sort_values(by='best_score', ascending=False, ignore_index=True)

    selected = available_ms[available_ms['name'].apply(lambda x: all([fil in x for fil in keywords]))]
    return selected

def get_class_weights(df):
    """
    returns map of class weights 
    """
    l = len(df.index)
    counts = df['label'].value_counts()
    return (df, {0: float(counts[True])/l, 1: float(counts[False])/l})

def _get_path(file_path, path_type):
    if type(file_path) is dict:
        file_path = file_path[path_type]
    return file_path

def best_score(log_file_path, metric=('val_cohen_kappa', 'max')):
    """
    metric: tuple ('str metric', 'max'/'min')
    
    returns best score or +/-inf depending on the optimum
    """
    try:
        data = pd.read_csv(_get_path(log_file_path, 'log'))[metric[0]]
        return data.max() if metric[1] == 'max' else data.min()
    except:
        return -np.Inf if metric[1] == 'max' else np.Inf

def last_epoch(log_file_path):
    """
    returns last epoch of training process based on last index in log
    """
    try:
        data = pd.read_csv(_get_path(log_file_path, 'log'))
        return int(data.iloc[-1]['epoch'])
    except:
        return -1
    
def set_note(note_file_path, note, write_mode = 'w', summary_append=None):
    """
    Sets note for the model
    """
    
    if summary_append is not None:
        #Take from https://stackoverflow.com/a/53668338
        stringlist = []
        summary_append.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        note = note + '\n--------------------\n' + short_model_summary

    with open(_get_path(note_file_path, 'note'), write_mode) as file: 
        file.write(note) 

def get_note(note_file_path):
    """
    Get note for the model
    """
    with open(_get_path(note_file_path, 'note'), "r+") as file:
        return file.read()
        
def get_log(path):
    return pd.read_csv(_get_path(path,'log'))

def model_image_shape(model):
    """
    returns shape of images accepted by model
    """
    shape = model.layers[0].input_shape[0]
    return (shape[1], shape[2])