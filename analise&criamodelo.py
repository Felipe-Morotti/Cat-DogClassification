import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import json

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing import image_dataset_from_directory

import os
import matplotlib.image as mpimg

#Visualizing the data: extract image paths and load them using Matplotlib.

fig = plt.gcf()
fig.set_size_inches(16, 16) 

cat_dir = os.path.join('dog-vs-cat-classification/cats')
dog_dir = os.path.join('dog-vs-cat-classification/dogs')
cat_names = os.listdir(cat_dir)
dog_names = os.listdir(dog_dir)

pic_index = 210

cat_images = [os.path.join(cat_dir, fname)  #select images based on pic_index.
              for fname in cat_names[pic_index-8:pic_index]]

dog_images = [os.path.join(dog_dir, fname)  #select images based on pic_index.
              for fname in dog_names[pic_index-8:pic_index]]

for i, img_path in enumerate(cat_images + dog_images):
    sp = plt.subplot(4, 4, i+1)   #creates a 4x4 grid for images.
    sp.axis('Off')   #hides the axis.

    img = mpimg.imread(img_path)   #reads each image and plt.imshow(img) displays it.
    plt.imshow(img)

plt.show()

#Splitting Dataset into training and validation sets.

base_dir = 'dog-vs-cat-classification'

train_datagen = image_dataset_from_directory(base_dir,
                image_size=(200,200),
                subset='training',
                seed = 1,
                validation_split=0.1,
                batch_size=32)

test_datagen = image_dataset_from_directory(base_dir,
                image_size=(200,200),
                subset='validation',
                seed=1,
                validation_split=0.1,
                batch_size=32)



#Model Architecture:
#Conv2D layers: extract image features like edges, shapes and textures.
#MaxPooling2D: reduces image dimensions while retaining important information.
#BatchNormalization: helps stabilize training and speed up convergence.
#Dropout layers: prevent overfitting.
#sigmoid activation: outputs a binary classification as Cat or Dog.

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

#model.summary()

#Model Compilation and Training:

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(train_datagen,
            epochs=5,
            validation_data=test_datagen)

with open('historico_treino.json', 'w') as f:
    json.dump(history.history, f)
    
model.save('meu_modelo.keras')

