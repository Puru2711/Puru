from simple_unet_model import simple_unet_model   #Use normal unet model
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt




def unettrain():
    image_directory = 'dataset/images/train/'
    mask_directory = 'dataset/images/mask/'


    SIZE = 256
    image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
    mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

    images = os.listdir(image_directory)
    for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
        if (image_name.split('.')[1] == 'png'):
            #print(image_directory+image_name)
            image = cv2.imread(image_directory+image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            
            image_dataset.append(np.array(image))

    #Iterate through all images in Uninfected folder, resize to 64 x 64
    #Then save into the same numpy array 'dataset' but with label 1

    masks = os.listdir(mask_directory)
    for i, image_name in enumerate(masks):
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(mask_directory+image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            mask_dataset.append(np.array(image))

    print(image_dataset)
    image_dataset = np.expand_dims(image_dataset, axis=3)
    image_dataset = image_dataset.astype('float32')


    mask_dataset = np.array(mask_dataset, np.float64) / 255.


    print("s")
    from sklearn.model_selection import train_test_split



    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size =0.1, random_state = 0)

    #Sanity check, view few mages
    import random
    import numpy as np
    image_number = random.randint(0, len(X_train))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
    plt.show()




    ###############################################################
    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH  = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]

    def get_model():
        return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = get_model()


    #If starting with pre-trained weights. 
    #model.load_weights('mitochondria_gpu_tf1.4.hdf5')
    
    history = model.fit(X_train, y_train, 
                        batch_size = 10, 
                        verbose=1, 
                        epochs=1, 
                        validation_data=(X_test, y_test), 
                        shuffle=False)

    model.save(r'C:\Users\madhu\gui_for_project\training_models\unet\unettrain.hdf5')

    ############################################################


    # evaluate model
    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy = ", (acc * 100.0), "%")
