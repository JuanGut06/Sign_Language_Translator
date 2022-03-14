from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as k
from random import randrange
import matplotlib.pyplot as plt
import cv2

k.clear_session()

training_data = 'D:/User/Project/Dataset/39words/Validacion'
validation_data = 'D:/User/Project/Dataset/39words/Validacion'

# Global variables
epochs = 3000
height, width = 200, 200
batch_size = 32
# Count of pictures over the batch size
steps = 1000/batch_size
steps_validation = 1000/batch_size

# Preparing pictures
data_generator_process = ImageDataGenerator(rescale=1./255, zoom_range=0.3)

# Preparing validation pictures
validation_processing = ImageDataGenerator(rescale=1./255)

data_gen_training = data_generator_process.flow_from_directory(
    training_data,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
)

validation_image = validation_processing.flow_from_directory(
    validation_data,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical'
)

# convolutional network with 3 layers
filter_32 = 32
filter_64 = 64
filter_128 = 128
filter_256 = 256
size_filter = (3, 3)
pool_size = (2, 2)
classes = 39
lr = 0.0005

# Create neuronal convolutional network
cnn = Sequential()

cnn.add(Convolution2D(filter_32, size_filter, padding='same', input_shape=(height, width, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(filter_64, size_filter, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(filter_128, size_filter, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(filter_256, size_filter, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Flatten())
# 256 for 2 classes
cnn.add(Dense(4992, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation='softmax'))

# Create tensorboard
resultBoard = TensorBoard(log_dir="logs/board_results")

# Compile and create the model
cnn.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

cnn.fit_generator(
    data_gen_training,
    steps_per_epoch=steps,
    epochs=epochs,
    validation_data=validation_image,
    validation_steps=steps_validation,
    callbacks=[resultBoard]
)

cnn.save('model.h5')
cnn.save_weights('pesos.h5')