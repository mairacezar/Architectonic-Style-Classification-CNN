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
        
# image size parameters
image_size = 227
LR = 1e-3
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') 

output_classes = 3

image_label_index = 0
image_label = [1, 0, 0]

def image_label_change(image_label, index):
    for i in range(len(image_label)):
        image_label[i] = 0
    if(index < 3):
        image_label[index] = 1


training_data = []

for folder_name in os.listdir(training_data_directory):
    # Numpy vector to store images
    #imgs = np.empty((0, size_x, size_y, 3))
    
    folder_path = os.path.join(training_data_directory, folder_name)
    
    #tratando imagens
    for image in os.listdir(folder_path):
        if(image.endswith(".jpg")):
            img = Image.open(os.path.join(folder_path, image)).convert('RGB')
            img = img.resize((image_size, image_size), Image.ANTIALIAS)

            label = image_label
            training_data.append([np.array(img), np.array(label)])

    image_label_index += 1
    image_label_change(image_label, image_label_index)
            
shuffle(training_data)
np.save("outfile", training_data)


train = training_data[:-30]
test = training_data[-30:]

train_X = np.array([i[0] for i in train]).reshape(-1, image_size, image_size, 1) 
train_Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, image_size, image_size, 1) 
test_y = [i[1] for i in test]



############################################################################## LALLALALALALALALALA




            























































































# Return all images that he can find inside program filesystem
#for root, dirs, files in os.walk(os.getcwd()):
#    for file in files:
#        if(file.endswith(".jpg")):
#            print(file)