from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import scipy.integrate

# img = cv2.imread("C:\\Users\\AKASH\\Tenserflow\\first_demo\\training\\happy\\1.jfif")


# cv2.imshow("img",img)
# cv2.waitKey(0)



train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)


train_dataset = train.flow_from_directory('C:/Users/AKASH/Tenserflow/first_demo/training/',target_size=(200,200),batch_size= 3,class_mode='binary')

validation_dataset = train.flow_from_directory('C:/Users/AKASH/Tenserflow/first_demo/validation/',target_size=(200,200),batch_size= 3,class_mode='binary')


# print(train_dataset.class_indices)
# print(train_dataset.classes)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),tf.keras.layers.MaxPool2D(2,2),
#
tf.keras.layers.Conv2D(32,(3,3),activation='relu'),tf.keras.layers.MaxPool2D(2,2),
#
tf.keras.layers.Conv2D(64,(3,3),activation='relu'),tf.keras.layers.MaxPool2D(2,2),
#
tf.keras.layers.Flatten(),
##
tf.keras.layers.Dense(512,activation='relu'),
##
tf.keras.layers.Dense(1,activation='sigmoid')
])


model.compile(loss= 'binary_crossentropy',
              optimizer= RMSprop(lr=0.001),
              metrics=['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch= 3,
                      epochs= 30,
                      validation_data=validation_dataset)


dir_path = 'C:/Users/AKASH/Tenserflow/first_demo/testing'


img = cv2.imread("C:\\Users\\AKASH\\Tenserflow\\first_demo\\testing\\happy\\_medium.jpeg").reshape(200,200)

cv2.imshow("img",img)
cv2.waitKey(0)