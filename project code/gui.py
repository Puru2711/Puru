# This will import all the widgets
# and modules which are available in
# tkinter and ttk module
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk,Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

#imgname=filedialog.askopenfilename()

#imgname=Image.open(s)
#imgname="2.png"


# creates a Tk() object
master = Tk()

# sets the geometry of main
# root window
master.geometry("300x300")
def vgggmodel():
    from vgg16modelsummary import m1
    mo=m1()
    print("Wait model is loading")
    mo.summary()

def vgggtrain():
    try:
        from vgg16train.train import vggtrain
        mo=vggtrain()
        if mo==True:
            print("model trained")
        else:
            print("model is not trained")
    except:
        print("Model is not loaded Successfully or there might be some error. Please re run the program and train")

def vgggtest():

    try:
        
        from vgg16test import t1
        imgname=filedialog.askopenfilename()
        result=t1(r"C:\Users\madhu\gui_for_project\trained_models\vgg16model\stmodel.h5",imgname)
        if result[0][1]>=0.4619:
            a='This Image may be Forged.Please segment the image for confirmation.'
        else:
            a='Not-Forged'

        print(a)
    except:
        print("Model is not loaded successfully or there might be some error. Please re run the program and test with Image.")

    
def vggg16():    
    newmaster1 = Tk()
    newmaster1.geometry("500x500")
    newmaster1.configure(bg='blue')
    label = Label(newmaster1,text ="VGG16 WINDOW")
    label.pack(pady = 10)
    btn = Button(newmaster1,text ="VGG16 MODEL",command = vgggmodel)
    btn.pack(pady = 10)
    btn = Button(newmaster1,text ="VGG16 TRAIN",command =vgggtrain)
    btn.pack(pady = 10)
    btn = Button(newmaster1,text ="VGG16 TEST",command = vgggtest)
    btn.pack(pady = 10)

def semanticsegmentationmodel():
    from simple_unet_model_summary import simple_unet_model
    smodel=simple_unet_model(256,256,3)
    print("Model is loading")
    smodel.summary()
    

def semanticsegmentationtrain():
    print("model is training")
    from unet.unettrain import unettrain
    result=unettrain()
    #print(result)


  

    
def semanticsegmentationtest():
    global testno
    imgname=filedialog.askopenfilename()
    from unettest import unettest
    result=unettest(imgname)
    #print(result)

    g=plt.figure(figsize=(8, 5))
    g.patch.set_facecolor('lightcyan')
    plt.subplot(121)
    x=500
    plt.title('Selected Image')
    #imgname1=mpimg.imread(imgname)
    imgname1=cv2.imread(imgname,cv2.COLOR_BGR2RGB)
    imgname1=cv2.cvtColor(imgname1,cv2.COLOR_BGR2RGB)
    imgname1=cv2.resize(imgname1,(x,x))
    plt.imshow(imgname1)
    plt.subplot(122)
    plt.title('Prediction of selected Image')
    result=cv2.resize(result,(x,x))
    plt.imshow(result, cmap='gray')
    plt.imsave(r"C:\Users\madhu\gui_for_project\output_images\unetmaskoutput{}.png".format(testno), result, cmap='gray')
    #plt.imsave(r"C:\Users\madhu\gui_for_project\output_images\orginalimage{}.png".format(testno),imgname, cmap='gray')
    plt.show()
    print("semanticsegmentationtest")


def semanticsegmentation():
    print("Semantic Segmentation using UNET")
    newmaster2 = Tk()
    newmaster2.geometry("500x500")
    newmaster2.configure(bg='blue')
    label = Label(newmaster2,text ="Semantic Segmentation Using UNET")
    label.pack(pady = 10)
    btn = Button(newmaster2,text ="UNET SUMMARY",command = semanticsegmentationmodel)
    btn.pack(pady = 10)
    btn = Button(newmaster2,text ="UNET TRAIN",command =semanticsegmentationtrain)
    btn.pack(pady = 10)
    btn = Button(newmaster2,text ="UNET TEST",command = semanticsegmentationtest)
    btn.pack(pady = 10)


def vgggsemanticsegmentationmodel():
    from semvgg16model import semanticvgg16model
    print("Modified vgg16 with 2layers for segmentation.")
    semanticvgg16model()
    

def vgggsemanticsegmentationtrain():
    from semanticvgg.vggsemsegtrain import vggsemsegtrain
    result=vggsemsegtrain()
    print("Training Completed")

def vgggsemanticsegmentationtest():
    from semtest import semvgg16
    imgname=filedialog.askopenfilename()
    result=semvgg16(imgname)
    #print(result)

    g=plt.figure(figsize=(8, 5))
    g.patch.set_facecolor('lightcyan')
    plt.subplot(121)
    x=500
    plt.title('Selected Image')
    #imgname1=mpimg.imread(imgname)
    imgname1=cv2.imread(imgname,cv2.COLOR_BGR2RGB)
    imgname1=cv2.cvtColor(imgname1,cv2.COLOR_BGR2RGB)
    imgname1=cv2.resize(imgname1,(x,x))
    plt.imshow(imgname1)
    plt.subplot(122)
    plt.title('Prediction of selected Image')
    result=cv2.resize(result,(x,x))
    plt.imshow(result, cmap='gray')
    plt.imsave(r"C:\Users\madhu\gui_for_project\output_images\vgg16segmaskoutput{}.png".format(testno), result, cmap='gray')
    #plt.imsave(r"C:\Users\madhu\gui_for_project\output_images\orginalimage{}.png".format(testno),imgname, cmap='gray')
    #cv2.imwrite(r"C:\Users\madhu\gui_for_project\output_images\vgg16segmaskoutput",result)
    plt.show()



def vgggsemanticsegmentation():
    newmaster3 = Tk()
    newmaster3.configure(bg='blue')
    newmaster3.geometry("500x500")
    label = Label(newmaster3,text ="VGG16 SEMANTIC SEGMENTATION WINDOW")
    label.pack(pady = 10)
    btn = Button(newmaster3,text ="MODEL SUMMARY",command =  vgggsemanticsegmentationmodel)
    btn.pack(pady = 10)
    btn = Button(newmaster3,text ="MODEL TRAINING",command =  vgggsemanticsegmentationtrain)
    btn.pack(pady = 10)
    btn = Button(newmaster3,text ="MODEL TEST",command =  vgggsemanticsegmentationtest)
    btn.pack(pady = 10)
    

def openNewWindow():
    newmaster = Tk()
    newmaster.geometry("500x500")
    newmaster.configure(bg='blue')
    label = Label(newmaster,text ="ALGORITHMS WINDOW")
    
    label.pack(pady = 10)
    btn = Button(newmaster,text ="VGG16",command = vggg16)
    
    btn.pack(pady = 10)
    btn = Button(newmaster,text ="SEMANTIC SEGMENTATION USING UNET",command = semanticsegmentation)
    btn.pack(pady = 10)
    btn = Button(newmaster,text ="SEMANTIC SEGMENTATION USING VGG16",command = vgggsemanticsegmentation)
    btn.pack(pady = 10)
    
def total():
    from vgg16test import t1
    imgname=filedialog.askopenfilename()
    result1=t1(r"C:\Users\madhu\gui_for_project\trained_models\vgg16model\stmodel.h5",imgname)
    print(result1)
    if result1[0][1]>=0.4619:
        a='FORGED'
    else:
        a='NOT-FORGED'



    from semtest import semvgg16
    result=semvgg16(imgname)
    

    g=plt.figure(figsize=(8, 5))
    g.patch.set_facecolor('lightcyan')
    plt.subplot(121)
    x=500
    plt.title('Selected Image')
    #imgname1=mpimg.imread(imgname)
    imgname1=cv2.imread(imgname,cv2.COLOR_BGR2RGB)
    imgname1=cv2.cvtColor(imgname1,cv2.COLOR_BGR2RGB)
    imgname1=cv2.resize(imgname1,(x,x))
    plt.imshow(imgname1)
    plt.subplot(122)
    plt.title('Prediction of selected Image as {}'.format(a))
    result=cv2.resize(result,(x,x))
    plt.imshow(result, cmap='gray')
    plt.show()







#testno=int(input("Please Enter Test no for saving images in folder with different number"))
testno=1
label = Label(master,text ="MAIN WINDOW")
master.configure(bg='blue')
label.pack(pady = 10)

# a button widget which will open a
# new window on button click
btn = Button(master,text ="TEST WITH EACH ALGORITHM",command = openNewWindow)
btn.pack(pady = 10)

btn = Button(master,text ="FINAL TEST ",command =total)
btn.pack(pady = 10)

# mainloop, runs infinitely
mainloop()
