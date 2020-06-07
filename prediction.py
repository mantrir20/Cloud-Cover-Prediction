import numpy as np
import tensorflow as tf
from PIL import Image
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import pylab as pl
import time
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.layers import MaxPooling2D, Dense, Flatten
from skimage.measure import compare_ssim
import imutils
import cv2
from google.colab.patches import cv2_imshow

#Loading array
SEQUENCE = np.load('/content/sample_data/sequence_array.npz')['sequence_array']  # load array
n_samples = len(SEQUENCE)

WIDTH = 100
HEIGHT = 100
n_frames = 2

# step =1
SEQUENCE = SEQUENCE.reshape(n_samples, WIDTH, HEIGHT, 1)
BASIC_SEQUENCE = np.zeros((n_samples-n_frames, n_frames, WIDTH, HEIGHT, 1))
NEXT_SEQUENCE = np.zeros((n_samples-n_frames, n_frames, WIDTH, HEIGHT, 1))

for i in range(n_frames):
    BASIC_SEQUENCE[:, i, :, :, :] = SEQUENCE[i:i+n_samples-n_frames]
    NEXT_SEQUENCE[:, i, :, :, :] = SEQUENCE[i+1:i+n_samples-n_frames+1]


### Create model:
### first layer has input shape of (nmovies, width, height, channels)
### returns identical shape
model=Sequential()
model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(None, WIDTH, HEIGHT, 1), padding='same', return_sequences=True))
model.add(BatchNormalization())
#model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3), padding='same', return_sequences=True))
#model.add(BatchNormalization())
model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
model.compile(loss='binary_crossentropy', optimizer='adadelta',metrics=['accuracy'])

### Split data into training and test sets
trainingfraction = 0.9
train_size = round(n_samples * trainingfraction)

### Train the network
model.fit(BASIC_SEQUENCE[:train_size], NEXT_SEQUENCE[:train_size], batch_size=10, epochs=13, validation_split=0.05)


### Take an example from the test set and predict the next steps
index = 7

num_test_frames = 1 ### Number of frames to predict


train_pred = BASIC_SEQUENCE[index][:n_frames-num_test_frames:, ::, ::, ::]  ##(track)
for j in range(n_frames):
    new_pos = model.predict(train_pred[np.newaxis, :, :, :, :])
    new = new_pos[:, -1, :, :, :]
    train_pred = np.concatenate((train_pred, new), axis=0)

### Compare predictions to the truth
truth = BASIC_SEQUENCE[index][:, :, :, :]

for i in range(n_frames):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ### In left panel show original then predicted frames
    if i >= (n_frames-num_test_frames):
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Inital trajectory', fontsize=20)
    toplot = train_pred[i, ::, ::, 0]
    plt.imshow(toplot, cmap='binary')
    plt.savefig('%i_Prediction.png' % (i + 1))

for i in range(n_frames):   
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)
    toplot = truth[i, ::, ::, 0]
    if i >= 2:
        toplot = NEXT_SEQUENCE[which][i - 1, ::, ::, 0]
    plt.imshow(toplot, cmap='binary')
    plt.savefig('%i_GroundTruth.png' % (i + 1))
