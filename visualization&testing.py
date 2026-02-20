import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

from tensorflow import keras
from keras.utils import image_dataset_from_directory
from keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

model = tf.keras.models.load_model('meu_modelo.keras')

with open('historico_treino.json', 'r') as f:
    history_dict = json.load(f)

#Model Evaluation: visualize the training and validation accuracy with each epoch.

history_df = pd.DataFrame(history_dict) #converts the training history into a DataFrame.
history_df.loc[:, ['loss', 'val_loss']].plot() #plots the training and validation loss.
history_df.loc[:, ['accuracy', 'val_accuracy']].plot() #plots the training and validation accuracy.
#plt.show()


#Model Testing and Prediction:

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(200, 200))
    plt.imshow(img)
    
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)
    print("Dog" if result >= 0.5 else "Cat")
    plt.show()

predict_image('dog-vs-cat-classification/cats/cat.4001.jpg')
predict_image('dog-vs-cat-classification/dogs/dog.4001.jpg')