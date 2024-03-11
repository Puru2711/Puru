import numpy as np 
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.preprocessing.image import ImageDataGenerator
import os

from keras.models import load_model
from numpy import loadtxt

def semanticvgg16model():
        VGG_model=load_model(r'C:\Users\madhu\gui_for_project\trained_models\vggsemseg\stmodel.h5')
        
        for layer in VGG_model.layers:
                layer.trainable = False
        new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
        new_model.summary()

