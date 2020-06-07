import numpy as np
from PIL import Image
import os
import time
import glob
from PIL import *
import skimage

IMAGE_PATH = '/content/'

WIDTH = 100
HEIGHT = 100
SEQUENCE = np.array([])
BASIC_SEQUENCE = np.array([])
NEXT_SEQUENCE = np.array([])
NUMBER = 0

def image_initialize(image):
    picture = Image.open(image)
    picture = picture.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    picture = picture.convert('L')
    picture.save('/content/sample_data/temp.png')  
    data = np.array(picture.getdata()).reshape(WIDTH, HEIGHT, 1)
    return data

for file in glob.glob('/content/'+"/*"+'.png'):
    image_array = (image_initialize(os.path.join(IMAGE_PATH, file)))/255
    SEQUENCE = np.append(SEQUENCE, image_array)
    NUMBER += 1
    print(SEQUENCE[0])
    print(NUMBER)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH * HEIGHT)

np.savez('/content/sample_data/sequence_array.npz', sequence_array=SEQUENCE)
print('Data saved.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

