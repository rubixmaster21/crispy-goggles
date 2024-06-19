# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:18:59 2024

@author: uyy
"""

import os
import keras
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

# Do not Load MNIST dataset
#(train_ds, _), (val_ds, _) = mnist.load_data()

batch_size = 600
img_height = 36
img_width = 36
data_dir = ".\\image_3D_png_tiled\\train\\"

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  color_mode='rgb',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  labels = None)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  color_mode='rgb',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  labels = None)


# Preprocess images
# train_ds = train_ds.reshape(-1, 784) / 255.0
# val_ds = val_ds.reshape(-1, 784) / 255.0
normalization_layer = tf.keras.layers.Rescaling(1./255)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# train_ds = tf.data.Dataset.from_tensor_slices(
#     tf.expand_dims(tf.concat([train_ds, val_ds], axis=0), axis=-1))

# train_ds = train_ds.take(int(1e4)).batch(4).map(lambda x: (x/255, x/255))



# Encoder definition
encoder = Sequential([
    Dense(128, activation='relu', input_shape=(36,36,3,)),
    Dense(64, activation='relu')
])
# We can compile the model here, otherwise there will be a warning when it is
# used, but it is optional.

# Decoder definition
decoder = Sequential([
    Dense(128, activation='relu', input_shape=(36,36,64,)),
    Dense(1296, activation='sigmoid')
])
# We can compile the model here, otherwise there will be a warning when it is
# used, but it is optional.

# Autoencoder definition
autoencoder = Sequential([
    encoder,
    decoder
])

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(train_ds, epochs=10, batch_size=256)

# Evaluation
loss = autoencoder.evaluate(val_ds, val_ds, verbose=0)
print("Test Loss:", loss)

# Save models
encoder_model_path = 'encoder_model.h5'
decoder_model_path = 'decoder_model.h5'

encoder.save(encoder_model_path)
decoder.save(decoder_model_path)

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

encoder_model_size = get_file_size(encoder_model_path)
decoder_model_size = get_file_size(decoder_model_path)
print("Encoder model size:", encoder_model_size)
print("Decoder model size:", decoder_model_size)

# Load the models
loaded_encoder = keras.models.load_model('encoder_model.h5')
loaded_decoder = keras.models.load_model('decoder_model.h5')

# Use loaded models for prediction
#
# In a real deployment, the encoder will be on the server side, the decoder will
# be on the client side, and the encoded images are what's transferred from the
# server to the client.
encoded_imgs = loaded_encoder.predict(val_ds)
decoded_imgs = loaded_decoder.predict(encoded_imgs)

# Visualization
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(val_ds[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
