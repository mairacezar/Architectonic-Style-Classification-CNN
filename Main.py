#https://stackoverflow.com/questions/50753668/how-do-i-set-up-a-image-dataset-in-tensorflow-for-a-cnn
#https://stackoverflow.com/questions/44808812/how-to-prepare-a-dataset-of-images-to-train-and-test-tensorflow
#https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
#https://github.com/MuhammedBuyukkinaci/TensorFlow-Multiclass-Image-Classification-using-CNN-s/blob/master/multiclass_classification_gpu.py

import os
import cv2
from PIL import Image
import numpy as np
#import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
#from sklearn.metrics import roc_auc_score

# Um dia tem que mudar isso para uma coisa menos confusa. 
program_folder = os.getcwd()
training_data_directory = os.path.join(program_folder, "Training Data")

        
# Defines
IMAGE_SIZE = 227
LR = 1e-3
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') 


#image_label_index = 0
#image_label = [1, 0, 0]

# Função do tipo Gambiarra para ajudar a setar o label pela pasta
def image_label_change(image_label, index):
    for i in range(len(image_label)):
        image_label[i] = 0
    if(index < 3):
        image_label[index] = 1

def get_train_data(data_directory):
    
    training_data = []
    
    # Acho que é necessário mudar a forma como estamos fazendo o label para uma
    # forma mais simples, expansível e "correta". Algo como
    # image_label = ["Label", "*label_name*"]
    image_label_index = 0
    image_label = [1, 0, 0]
    
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
                label = image_label
                
                # Cria o dataset no formato 
                # DATA [numpy_array_label, numpy_array_feature]
                training_data.append([np.array(img), np.array(label)])
                
        # Gambiarra para fazer o label mudar automativamente quando muda de pasta
        image_label_index += 1
        image_label_change(image_label, image_label_index)
    
    # Embaralhando os dados            
    shuffle(training_data)
    
    # Salvando o dataset como arquivo .npy
    #np.save("outfile", training_data)
    
    # Retornando o dataset
    return training_data

# Autoexplicativo
data = get_train_data(training_data_directory)

# Divide o dataset em train e test (272 e 30 dados cada)
train = data[:-30]
test = data[-30:]

# Teoricamente cria o vetor de Features e Labels para o Train e o Test
train_X = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) 
train_Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) 
test_y = [i[1] for i in test]

# Função importantíssima do código que mostra para o seu programador quando o
# código terminou.
print("end program")

############################################################################## LALLALALALALALALALA <----- Não pode, passou
#                                                                                                         dos 80 caracteres




            






















































































# Quem sabe um dia agente usa isso...
# Return all images that he can find inside program filesystem
#for root, dirs, files in os.walk(os.getcwd()):
#    for file in files:
#        if(file.endswith(".jpg")):
#            print(file)