# Bird Classification
This is a bird image classifier using python and machine learning. 

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Linxi Wei | The Affiliated High School of Peking University | Computer Science | Incoming Senior |

![image](https://user-images.githubusercontent.com/79397351/128707285-b6f6c0c2-5881-4d87-88a6-5ebf65c6f8a6.png)

# First Milestone
My first milestone is getting to know some basic knowledge in machine learning, setting up the Raspberry Pi, and finding a dataset for birds. 

## 1.Machine learning 
Machine learning is like the opposite process of traditional programing. In machine learning, the computer should figure out the rules based on the answers and data you gave it. 

For a start, I learned Tensorflow, a library created by Google to implement machine learning models, and built a sample program that can recognise numbers in mnist. Here is the code:

```python
import tensorflow as tf
import matplotlib.pyplot as plt
mnist=tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()
model=tf.keras.models.Sequential([
 tf.keras.layers.Flatten(input_shape=(28,28)),
 tf.keras.layers.Dense(128,activation='relu'),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(10, activation='softmax')
 ]) 
 model.summary()
 model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 model.fit(image_train, label_train, epochs=5)

 model.evaluate(image_test, label_test, verbose=2)
```
## 2.Setting up the Raspberry Pi
I successfully connected the Raspberry Pi with the monitor, the keyboard, and the mouse. Now it is ready to recieve my model and start working. 

![image](https://user-images.githubusercontent.com/79397351/128520390-fa537a86-9fef-4337-b342-8dc4a421f0e2.png)


## 3.Dataset
I found a very good dataset for bird classification (https://www.kaggle.com/gpiosenka/100-bird-species). It contains 275 bird species——39364 training images, 1375 test images(5 per species), and 1375 validation images. 

I uploaded the dataset to Google Colab from Google Drive and did some data pre-processing work. For the next step, I will transform the data and build my classifier. I decided to use Pytorch and the pre-trained VGG16 model.

Useful codes to upload dataset:
```python
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/gdrive', force_remount=True)
```
<iframe width="560" height="315" src="https://www.youtube.com/embed/4Fx8QDezMp4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Second Milestone
My second milestone is building up my machine learning model. Since the dataset I chose is too big, I randomly selected 20 bird species to form my training array. I used vgg16 as my base model and built my own model on top of it. My final model is like this:

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
softmax (Dense)              (None, 20)                501780    
=================================================================
Total params: 15,216,468
Trainable params: 501,780
Non-trainable params: 14,714,688
_________________________________________________________________
```
After building the model, I began to train it. I did training and testing at the same time, and I used 5 epochs. To make it done more quickly, I used GPU to run it. 

# Final Milestone
