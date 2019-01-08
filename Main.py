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
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


#_________________________CONSTRUCT DATASET___________________________________


# Um dia tem que mudar isso para uma coisa menos confusa. 
program_folder = os.getcwd()
training_data_directory = os.path.join(program_folder, "Training Data")

        
# Defines
IMAGE_SIZE = 227


def label_images(image):
    if("ba" in image):
        return np.array([1,0,0])
    elif("go" in image):
        return np.array([0,1,0])
    elif("ro" in image):
        return np.array([0,0,1])

# Function to create the dataset
def get_train_data(data_directory):
    
    training_data = []
    
    # Entrando na pasta das pastas das images
    for folder_name in os.listdir(data_directory):       
        folder_path = os.path.join(data_directory, folder_name)
        
        # Tratando imagens
        for image in os.listdir(folder_path):
            if(image.endswith(".jpg")):
                # Abre a imagem e converte ela para RGB
                img = Image.open(os.path.join(folder_path, image)).convert('RGB')
                
                # Resize a imagem para 227,227. Até onde sabemos essa operção 
                # não é muito destrutiva
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                
                # Mais uma parte da gambiarra de setar o label
                label = label_images(image)
                
                # Cria o dataset no formato 
                # DATA [numpy_array_label, numpy_array_feature]
                training_data.append([np.array(img), label])
                
    # Embaralhando os dados            
    shuffle(training_data)
    
    # Salvando o dataset como arquivo .npy
    #np.save("outfile", training_data)
    
    # Retornando o dataset
    return training_data


# Function to load dataset file if exist or create a new one if doesn't
def load_npy_if_exists( directory ):
    # Flag to check if file was found
    file_exists = False
    
    # For each file in file root dir, check if an .npy file exists, if it does
    # load it, if it does not, create a dataset file and saves it as dataset.npy
    for file in os.listdir():
        if ( file.endswith( ".npy" ) ):
            print("Loading .npy from folder...")
            return( np.load( file ) )
            file_exists = True
    if ( file_exists == False ):
        print("Creating dataset from images...")
        dataset = get_train_data( directory )
        print("Dataset Created, now saving...")
        np.save("dataset", data)
        print("Saved dataset as: dataset.npy with succes!")
        return(dataset)


# Old Get_data func, runs everytime.
#data = get_train_data(training_data_directory)

# Check if folder already has .npy datase, if it does, loads it, if not, creates
# and saves it as an .npy file.
# Checa se a pasta ja tem um dataset .npy, se tem, carrega o arquivo, se não,
# cria um dataset e salva ele como um arquivo .npy
data = load_npy_if_exists(training_data_directory)        


# Divides dataset into Train and test ( 272 and 30 data each)
# Divide o dataset em Train e Test (272 e 30 dados cada)
train = data[:-30]
test = data[-30:]

# Creates vectors for Features and Labels for Train and Test
# Cria um vetores para Features e um para Labels para os dados de Test e de Train
train_x = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3) 
train_y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3) 
test_y = np.array([i[1] for i in test])

# Função importantíssima do código que mostra para o seu programador quando o
# código terminou.
print("end program")


#________________________TESTING DATASET______________________________________

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_x.shape))

# Display the first image in training data
plt.imshow(train_x[0], cmap='gray')
plt.show()

# Print image label for sanity checks
print(train_y[0], train_y.shape)


#____________________CONSTRUCT NEURAL




































































# Quem sabe um dia agente usa isso...
# Return all images that he can find inside program filesystem
#for root, dirs, files in os.walk(os.getcwd()):
#    for file in files:
#        if(file.endswith(".jpg")):
#            print(file)