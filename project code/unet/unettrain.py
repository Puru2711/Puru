from unet.simple_unet_model import simple_unet_model   #Use normal unet model
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt



def unettrain():
    image_directory = r'C:\Users\madhu\gui_for_project\unet\dataset\images/train/'
    mask_directory =  r'C:\Users\madhu\gui_for_project\unet\dataset\images/mask/'


    SIZE = 256
    image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
    mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
    fco=0
    fli=[]
    mli=[]
    mco=0
    images = os.listdir(image_directory)
    import numpy as np
    for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
        if (image_name.split('.')[1] == 'png'):
            #print(image_directory+image_name)
            fli.append(image_name)
            fco+=1
            image = cv2.imread(image_directory+image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            
            image_dataset.append(np.array(image))

    #Iterate through all images in Uninfected folder, resize to 64 x 64
    #Then save into the same numpy array 'dataset' but with label 1
    masks = os.listdir(mask_directory)
    for i, image_name in enumerate(masks):
        if (image_name.split('.')[1] == 'png'):
            mli.append(image_name)
            mco+=1
            image = cv2.imread(mask_directory+image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            mask_dataset.append(np.array(image))
    print(fco)

    print("{}------------------------>{}".format("forged_image","Mask_image"))
    for i in range(0,fco):
        print("{}------------->{}".format(fli[i],mli[i]))



    
    image_dataset = np.expand_dims(image_dataset, axis=3)
    image_dataset = image_dataset.astype('float32')


    mask_dataset = np.array(mask_dataset, np.float64) / 255.


    from sklearn.model_selection import train_test_split


    
    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size =0.1, random_state = 0)



    
    ###############################################################
    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH  = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]

    def get_model():
        return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = get_model()


    #If starting with pre-trained weights. 
    #model.load_weights('mitochondria_gpu_tf1.4.hdf5')
    l=[]
    history = model.fit(X_train, y_train, 
                        batch_size = 10, 
                        verbose=1, 
                        epochs=1, 
                        validation_data=(X_test, y_test), 
                        shuffle=False)

    model.save(r'C:\Users\madhu\gui_for_project\training_models\unet\unettrain.hdf5')

    ############################################################


    # evaluate model
    _, trainacc = model.evaluate(X_train, y_train)
    _, testacc= model.evaluate(X_test, y_test)
    '''
    import matplotlib.pyplot as plt
    data={"Forged Image":fco,"Correspond Mask Image":mco}
    Imgcat=list(data.keys())
    Imgcou=list(data.values())
    fig=plt.figure(figsize=(3,3))
    plt.bar(Imgcat,Imgcou,color=['green','black'],width=0.5)
    plt.xlabel("Image Types")
    plt.ylabel("No of Images")
    plt.show()
    '''
    import matplotlib.pyplot as plt
    fig=plt.figure()
    fig.patch.set_facecolor('lightcyan')
    fig.suptitle("ACCURACY OF UNET MODEL\nTrain_Accuracy:{}\nTest_Accuracy{}:".format(trainacc,testacc),fontsize=20,fontweight='bold',color='lime')
    
    plt.show()
    
    
    return True

