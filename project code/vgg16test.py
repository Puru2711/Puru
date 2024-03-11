from keras.models import load_model
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def t1(mpath,imagep):
    
    model=load_model(mpath)

    from matplotlib.pyplot import imread
    from matplotlib.pyplot import imshow
    img_path=imagep
    img =image.load_img(img_path,target_size=(224,224))
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    
    #print('image',x.shape)

    my_image=imread(img_path)
    features=model.predict(x)
    li=features.tolist()
    return li
