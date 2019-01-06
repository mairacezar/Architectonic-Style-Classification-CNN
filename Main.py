#https://stackoverflow.com/questions/50753668/how-do-i-set-up-a-image-dataset-in-tensorflow-for-a-cnn
#https://stackoverflow.com/questions/44808812/how-to-prepare-a-dataset-of-images-to-train-and-test-tensorflow
#https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
#https://github.com/MuhammedBuyukkinaci/TensorFlow-Multiclass-Image-Classification-using-CNN-s/blob/master/multiclass_classification_gpu.py

import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


program_folder = os.getcwd()
training_data_directory = os.path.join(program_folder, "Training Data")
        
# image size
size_x = 80
size_y = 80


for folder_name in os.listdir(training_data_directory):
    # Numpy vector to store images
    imgs = np.empty((0, size_x, size_y, 3))
    
    folder_path = os.path.join(training_data_directory, folder_name)
    
    #tratando imagens
    for image in os.listdir(folder_path):
        if(image.endswith(".jpg")):
            print("banana")

for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if(file):
            print("Macaco")