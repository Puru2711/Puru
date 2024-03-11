import numpy as np
import warnings
import pandas as pd
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


def vggtrain():
    #from vgg16train.vgg16model import m1
    #m1()
    
    print("Model is loading wait")
    model=load_model(r"C:\Users\madhu\gui_for_project\training_models\stmodel.h5")
    dataset_path=os.listdir(r'C:\Users\madhu\gui_for_project\vgg16train\datasetf')
    room_types=os.listdir(r'C:\Users\madhu\gui_for_project\vgg16train\datasetf')
    
    rooms=[]
    for item in room_types:
        all_rooms=os.listdir(r'C:\Users\madhu\gui_for_project\vgg16train\datasetf'+'/'+item)
    
        for room in all_rooms:
            rooms.append((item,str(r'C:\Users\madhu\gui_for_project\vgg16train\datasetf'+'/'+item)+'/'+room))
            #print(rooms)
    
    
    rooms_df=pd.DataFrame(data=rooms,columns=['Image type','image'])
    print(rooms_df)
    
    
    
    print("Total Images:",len(rooms_df))
    room_count=rooms_df['Image type'].value_counts()
    print("Images in category:",room_count)
    
    import cv2
    path=r'C:\Users\madhu\gui_for_project\vgg16train\datasetf'
    
    img_size=224
    
    images=[]
    labels=[]
    tamp=0
    org=0
    
    for i in room_types:
        data_path=path +'/'+ str(i)
        filenames=[i for i in os.listdir(data_path)]
    
        for f in filenames:
            if i=='original':
                org+=1
            else:
                tamp+=1
            img=cv2.imread(data_path+'/'+f)
            img=cv2.resize(img,(img_size,img_size))
            images.append(img)
            labels.append(i)
    

    images=np.array(images)
    images=images.astype('float32')/255.0
    #images.shape

    #print(images)

    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    from sklearn.compose import ColumnTransformer
    y=rooms_df['Image type'].values
    #print(y)
    y_labelencoder=LabelEncoder()
    #print(len(y.shape))
    y=y_labelencoder.fit_transform(y)
    #print(y)
    
    y=y.reshape(-1,1)
    #print(y)
    ct=ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    y = np.array(ct.fit_transform(y))
    #print(y)


    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split

    images, Y=shuffle(images,y,random_state=1)
    train_x,test_x,train_y,test_y=train_test_split(images,y,test_size=0.05,random_state=415)

    model.fit(train_x,train_y,batch_size=2,epochs=1,verbose=0,)
    '''
    history=model.fit(train_x,train_y,batch_size=2,epochs=1,verbose=1,validation_data=(test_x,test_y))
    print(history.history)
    print(history.history['val_accuracy'])
    '''
    '''
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('plotting')
    plt.ylabel('acc loss')
    plt.xlabel('epoch')
    plt.legend(['acc','valacc','loss','valloss'],loc='upper left')
    plt.show()'''
    
    
    model.save(r"C:\Users\madhu\gui_for_project\training_models\stmodel.h5")
    _, trainacc = model.evaluate(train_x, train_y)
    _, testacc = model.evaluate(test_x, test_y)
    '''
    import matplotlib.pyplot as plt
    data={"ORIGINAL":org,"TAMPERED":tamp}
    Imgcat=list(data.keys())
    Imgcou=list(data.values())
    fig=plt.figure(figsize=(3,3))
    plt.bar(Imgcat,Imgcou,color=['blue','green'],width=0.5)
    plt.xlabel("Image Types")
    plt.ylabel("No of Images")
    plt.show()
    '''
    
    import matplotlib.pyplot as plt
    fig=plt.figure()
    fig.patch.set_facecolor('lightcyan')
    fig.suptitle("ACCURACY OF VGG16 MODEL\nTrain_Accuracy:{}\nTest_Accuracy{}:".format(trainacc,testacc),fontsize=20,fontweight='bold',color='lime')
    

    
    plt.show()
    
    #print("Accuracy = ", (acc * 100.0), "%")
    
    return True

