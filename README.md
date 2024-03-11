# Digital-Image-Forgery-Detection-Using-Convolutional-Neural-Network
Working demo video of project:https://youtu.be/CyCfmq3NwJ0

In my experience I would like to suggest to create dataset for image forgery projects. Implemnt this project in matlab beacause if you do in python you have to download dataset 
with corresponding mask images. If you do in matlab it will have inbuilt you can download normal classification casia dataset and you create labels in matlab. If you do in any python idle it will not have tha t feature. For me also , Faced a lot difficulties for dataset preparation.



This is the my minor project what i did in college.
I would like to suggest take it has reference beacause it will take time to implements this project for you guys. Because you may think different way of creating folder i think different way that why. And also you may use this code for creatng your own weights.

If I got time I will add some reference videos and websites. And my trained weigths. 

In Here I didn't added vgg16(classification and segmentation weights) because this algorithms having an approx of GB. I apploaded unet weigths If you want you can use.


In here I used three types of algorithms. 1st one is vgg16 in this project vgg16 is used for classsification and semantic segmentation. For getting good result I also applied 
random forest classifier. For segmentation using  vgg16 I implemented two layers of vgg16 and rf Classifer.
2nd algorihtm is Unet it get good accuracy compared to unet .
But vgg16 finally acheived good resullts . so in gui final test button I add these two algorithms(vgg16 classification and segmentation).

In segmentation white space represents forged part .
Accuracy and input and output of vgg16 classication.

![](https://github.com/Madhu11266/Digital-Image-Forgery-Detection-Using-Convolutional-Neural-Network/blob/learner/screenshots/Screenshot%202021-06-25%20140029.png)

Vgg16 segmentation using Random Forest classifier.

![](https://github.com/Madhu11266/Digital-Image-Forgery-Detection-Using-Convolutional-Neural-Network/blob/learner/screenshots/Screenshot%202021-06-25%20140221.png)

Unet for segmentation.

![](https://github.com/Madhu11266/Digital-Image-Forgery-Detection-Using-Convolutional-Neural-Network/blob/learner/screenshots/Screenshot%202021-06-25%20140321.png)


Final test I used the vgg16 for classification and segmentation(vgg16 only beacause for me it gives good results).

![](https://github.com/Madhu11266/Digital-Image-Forgery-Detection-Using-Convolutional-Neural-Network/blob/learner/screenshots/Screenshot%202021-06-25%20140340.png)
