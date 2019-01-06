#https://stackoverflow.com/questions/50753668/how-do-i-set-up-a-image-dataset-in-tensorflow-for-a-cnn
#https://stackoverflow.com/questions/44808812/how-to-prepare-a-dataset-of-images-to-train-and-test-tensorflow
#https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
#https://github.com/MuhammedBuyukkinaci/TensorFlow-Multiclass-Image-Classification-using-CNN-s/blob/master/multiclass_classification_gpu.py

import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm


# change to work on all computers
for directories in os.listdir():
    print (os.listdir())
    if directories == "Training data":
        print (directories)
    else:
        print ("banana")
    

# image size
size_x = 80
size_y = 80

