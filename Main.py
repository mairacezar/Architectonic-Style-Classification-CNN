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


# TODO: mudar isso para uma coisa menos confusa. 
program_folder = os.getcwd()
training_data_directory = os.path.join(program_folder, "Training Data")

        
# Defines.
IMAGE_SIZE = 227

# ENG: Returns a label based on image name
# PT: Retorna um label baseado no nome da imagem
def label_images(image):
    if("ba" in image):
        return np.array([1,0,0])
    elif("go" in image):
        return np.array([0,1,0])
    elif("ro" in image):
        return np.array([0,0,1])

# ENG: Function to create the dataset.
# PT: Função que cria um dataset.
def generate_dataset(data_directory):
    
    training_data = []
  
    # ENG: Entering the folder that contains all images folders.
    # PT: Entrando na pasta que contém todas as pastas das images.
    for folder_name in os.listdir(data_directory):       
        folder_path = os.path.join(data_directory, folder_name)
        
        # ENG: Treating images.
        # PT: Tratando imagens.
        for image in os.listdir(folder_path):
            if(image.endswith(".jpg")):
                # ENG: Opening images and converting them to RGB.
                # PT: Abre a imagem e converte ela para RGB.
                img = Image.open(os.path.join(folder_path, image)).convert('RGB')
                
                # ENG: Resizes images to 277,277.
                # PT: Resize a imagem para 227,227.
                # NOTE: Até onde sabemos essa operção não é muito destrutiva.
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                
                # ENG: Get a label for the image
                # PT: Adquire um label para a imagem
                label = label_images(image)
                
                # ENG: Creates a dataset and formats it to:
                # PT: Cria o dataset no formato de:
                # DATA [numpy_array_label, numpy_array_feature].
                training_data.append([np.array(img), label])
    
    # ENG: Shuffles data.
    # PT: Embaralha os dados.
    shuffle(training_data)
    
    # OLD: Salvando o dataset como arquivo .npy
    #np.save("outfile", training_data)
    
    # ENG: Returns dataset.
    # PT: Retorna o dataset.
    return training_data


# ENG: Function to load dataset file if exist or create a new one if doesn't.
# PT: Função para carregar o arquivo do dataset se ele existe, ou criar um novo
# se ele não existe.
def load_dataset_if_exists( directory ):
    # ENG: For each file in file root dir, check if an .npy file exists, if it does
    # load it, if it does not, create a dataset file and saves it as dataset.npy.
    # PT: Para cada arquivo presente no diretório root, checa se é um .npy, se
    # for, carrega o arquivo, se não, cria um arquivo dataset e salva ele como
    # dataset.npy.
    for file in os.listdir():
        if ( file.endswith( ".npy" ) ):
            print("Loading .npy from folder...")
            return( np.load( file ) )
    
    # ENG: If the dataset is not present in the folder, generates and them saves it.
    # PT: Se o dataset não está presente no arquivo, gera ele e salva.
    print("Generating dataset from images...")
    dataset = generate_dataset( directory )
    print("Dataset Created, now saving...")
    np.save("dataset", dataset)
    print("Saved dataset as: dataset.npy with succes!")
    return(dataset)


# OLD: Get_data func, runs everytime.
#data = get_train_data(training_data_directory)
    
    
# ENG: Check if folder already has .npy datase, if it does, loads it, if not, creates
# and saves it as an .npy file.
# PT: Checa se a pasta ja tem um dataset .npy, se tem, carrega o arquivo, se não,
# cria um dataset e salva ele como um arquivo .npy
data = load_dataset_if_exists(training_data_directory)        


# ENG: Divides dataset into Train and test ( 272 and 30 data each)
# PT: Divide o dataset em Train e Test (272 e 30 dados cada)
train = data[:-30]
test = data[-30:]

# ENG: Creates vectors for Features and Labels for Train and Test
# PT: Cria um vetores para Features e um para Labels para os dados de Test e de Train
train_x = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3) 
train_y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3) 
test_y = np.array([i[1] for i in test])

# NOTE: Função importantíssima do código que mostra para o seu programador quando o
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





















































# References

#https://stackoverflow.com/questions/50753668/how-do-i-set-up-a-image-dataset-in-tensorflow-for-a-cnn
#https://stackoverflow.com/questions/44808812/how-to-prepare-a-dataset-of-images-to-train-and-test-tensorflow
#https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
#https://github.com/MuhammedBuyukkinaci/TensorFlow-Multiclass-Image-Classification-using-CNN-s/blob/master/multiclass_classification_gpu.py






# Quem sabe um dia agente usa isso...
# Return all images that he can find inside program filesystem
#for root, dirs, files in os.walk(os.getcwd()):
#    for file in files:
#        if(file.endswith(".jpg")):
#            print(file)