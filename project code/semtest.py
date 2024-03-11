import pickle
import cv2
import numpy as np
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt



def semvgg16(imgpath):
    SIZE_X = 224 #Resize images (height  = X, width = Y)
    SIZE_Y = 224
    filename = r'C:\Users\madhu\gui_for_project\trained_models\vggsemseg\vggsemseg.sav'
    VGG_model=load_model(r'C:\Users\madhu\gui_for_project\trained_models\vggsemseg\stmodel.h5')
    new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
    
    #Load model.... 
    loaded_model = pickle.load(open(filename, 'rb'))
    
    #Test on a different image
    #READ EXTERNAL IMAGE...
    test_img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    t2=cv2.imread(imgpath,cv2.IMREAD_COLOR)
    t2= cv2.resize(t2, (SIZE_Y, SIZE_X))
    #print(type(test_img))
    #print(test_img)
    test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_img = np.expand_dims(test_img, axis=0)
    
    #predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
    X_test_feature = new_model.predict(test_img)
    X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])
    
    prediction = loaded_model.predict(X_test_feature)
    
    #View and Save segmented image
    prediction_image = prediction.reshape(224,224)
    return prediction_image
    '''
    plt.figure(figsize=(16, 8))
    plt.subplot(234)
    plt.title('External Image')
    plt.imshow(t2, cmap='gray')
    plt.subplot(235)
    plt.title('Prediction of external Image')
    plt.imshow(prediction_image, cmap='gray')
    plt.show()
    '''
