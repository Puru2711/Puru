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

def vggsemsegtrain():
    VGG_model=load_model(r'C:\Users\madhu\gui_for_project\training_models\vggsemseg\stmodel.h5')
    #print(os.listdir(r'C:\Users\madhu\gui_for_project\semanticvgg\images'))



    SIZE_X = 224 #Resize images (height  = X, width = Y)
    SIZE_Y = 224


    #Capture training image info as a list
    train_images = []

    for directory_path in glob.glob(r"C:\Users\madhu\gui_for_project\semanticvgg\images\train"):
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.resize(img, (SIZE_Y, SIZE_X))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            #train_labels.append(label)
    train_images = np.array(train_images)

    train_masks = [] 
    for directory_path in glob.glob(r"C:\Users\madhu\gui_for_project\semanticvgg\images\mask"):
        for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
            mask = cv2.imread(mask_path, 0)       
            mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
            #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            train_masks.append(mask)
            #train_labels.append(label)
    #Convert list to array for machine learning processing          
    train_masks = np.array(train_masks)


    X_train = train_images
    y_train = train_masks
    y_train = np.expand_dims(y_train, axis=3) #May not be necessary.. leftover from previous code 

    for layer in VGG_model.layers:
            layer.trainable = False
        
    #VGG_model.summary()  #Trainable parameters will be 0


    new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
    


    #Now, let us apply feature extractor to our training data
    features=new_model.predict(X_train)


    #Reassign 'features' as X to make it easy to follow
    X=features
    X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

    #Reshape Y to match X
    Y = y_train.reshape(-1)

    #print(Y)


    #Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
    #In our labels Y values 0 = unlabeled pixels. 
    dataset = pd.DataFrame(X)
    #print(dataset.values)

    dataset['Label'] = Y
    print(dataset['Label'].unique())
    print(dataset['Label'].value_counts())

    ##If we do not want to include pixels with value 0 
    ##e.g. Sometimes unlabeled pixels may be given a value 0.
    dataset = dataset[dataset['Label'] != 0]

    #Redefine X and Y for Random Forest
    X_for_RF = dataset.drop(labels = ['Label'], axis=1)
    Y_for_RF = dataset['Label']



    #RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 50, random_state = 42)

    # Train the model on training data
    model.fit(X_for_RF, Y_for_RF)

    #Save model for future use
    filename = r'C:\Users\madhu\gui_for_project\training_models\vggsemseg\vggsemsegmodel.sav'
    pickle.dump(model, open(filename, 'wb'))

    #Load model.... 
    #loaded_model = pickle.load(open(filename, 'rb'))
    a=True
    return a

