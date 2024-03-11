from simple_unet_model_summary import simple_unet_model 
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt



def unettest(imagepath):
    image_directory = 'unet/dataset/images/train/'
    mask_directory = 'unet/dataset/images/mask/'
    
    
    SIZE = 256
    image_dataset = []   
    mask_dataset = []  
    
    images = os.listdir(image_directory)
    for i, image_name in enumerate(images): 
        if (image_name.split('.')[1] == 'png'):
           #print(image_directory+image_name)
            image = cv2.imread(image_directory+image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            
            image_dataset.append(np.array(image))
    
    
    
    image_dataset = np.expand_dims(image_dataset, axis=3)
    image_dataset = image_dataset.astype('float32')
    
    
    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH  = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]

    def get_model():
        return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = get_model()
    
    
    
    model.load_weights(r"C:\Users\madhu\gui_for_project\trained_models\unet\unetmodel.hdf5")
    
    
    test_img_other = cv2.imread(imagepath, 0)
    test_img_other = cv2.resize(test_img_other, (SIZE, SIZE))
    
    test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
    test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
    test_img_other_input=np.expand_dims(test_img_other_norm, 0)
    
    
    prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)


    return prediction_other
    
    

    
