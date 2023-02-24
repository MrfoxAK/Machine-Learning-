
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.models import Sequential
import numpy as np
from glob import glob

IMAGE_SIZE = [224, 224]

train_path = 'C:/Users/AKASH/Tenserflow/2nd_demo/plant/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_path = 'C:/Users/AKASH/Tenserflow/2nd_demo/plant/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/test'

Inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet',include_top=False)


for layer in Inception.layers:
    layer.trainable = False

folders = glob('C:/Users/AKASH/Tenserflow/2nd_demo/plant/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/*')


x = Flatten()(Inception.output)



prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=Inception.input, outputs=prediction)


# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('C:/Users/AKASH/Tenserflow/2nd_demo/plant/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')



test_set = test_datagen.flow_from_directory('C:/Users/AKASH/Tenserflow/2nd_demo/plant/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


