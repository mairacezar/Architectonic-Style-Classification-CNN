import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from IPython import get_ipython
from random import randint
from skimage.util import random_noise
from skimage import exposure
get_ipython().run_line_magic('matplotlib', 'inline')

# TODO: Mais opões, alias, todas as opões possíveis
#n_iterations = int( input("Numeber of iterations: ") )
#e = int( input("Number of epochs: ") )

#def multiple_iterations(n_epochs):
   

#_________________________CONSTRUCT DATASET___________________________________#


# TODO: mudar isso para uma coisa menos confusa. 
# talvez usar root = os.path.abspath(__file__)
program_folder = os.getcwd()
#root = os.path.abspath(__file__)
training_data_directory = os.path.join(program_folder, "Training Data")
log_save_directory = os.path.join(program_folder, "Logs")

        
# Defines.
IMAGE_SIZE = 224
NUM_CLASSES = 3

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

def augmentation (data):
    
    aug_data = np.copy(data)
    #crop_size = IMAGE_SIZE
    
    for i in range(len(aug_data)-1):            
        case = randint(0,2)
        
        if (case == 1):
           aug_data[i][0] =  np.fliplr(aug_data[i][0])
           
        case = randint(0,1)    
        
        #if (case == 1):
            #aug_data[i][0] = random_noise(aug_data[i][0])

        case = randint(0,1)
    return (aug_data)

new_data = augmentation(data)

data = np.concatenate((data, new_data), axis=0)

#    plt.imshow(data[415][0])
#    plt.show()
#    print("lbl",data[415][1])
#    plt.imshow(data[389][0])
#    plt.show()
#    print("lbl",data[389][1])
#    plt.imshow(data[500][0])
#    plt.show()
#    print("lbl",data[500][1])

shuffle(data)
#print("data shape",np.shape(data[0][0]))




# ENG: Divides dataset into Train and test ( 272 and 30 data each)
# PT: Divide o dataset em Train e Test (272 e 30 dados cada)
train = data[:-60]
test = data[-60:]


# ENG: Creates vectors for Features and Labels for Train and Test
# PT: Cria um vetores para Features e um para Labels para os dados de Test e de Train
train_x = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3) 
train_y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3) 
test_y = np.array([i[1] for i in test])

#________________________TESTING DATASET______________________________________#

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_x.shape))

# Display the first image in training data
print(train_x[0].shape)

plt.imshow(train_x[415])
plt.show()
print("lbl",data[415][1])
plt.imshow(train_x[389])
plt.show()
print("lbl",data[389][1])
plt.imshow(train_x[500])
plt.show()
print("lbl",data[500][1])

print(train_x.shape)
print(train_y.shape)        

#    print("Train shape", train_y[0], train_y.shape)
#    for i in range(len(train_y)):
#        print(train_y[i])

#____________________CONSTRUCT CONVOLUTIONAL NEURAL NETWORK___________________#
    

        
#_______________________________ Logging _____________________________________#
#    
#    
#    import datetime
#    now = datetime.datetime.now()
#    
#    time = now.strftime("Log %d-%m-%Y - %H-%M-%S")
#    
#    os.makedirs( os.path.join( log_save_directory, time ) )
#    
#    log_folder = os.path.join(log_save_directory, time)
#    
#    
#    
#    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
#    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
#    
#    #axes1 = plt.gca()
#    #axes1.set_ylim([0,0.1])
#    
#    fig1 = plt.gcf()
#    plt.title('Training and Test loss')
#    plt.xlabel('Epochs ',fontsize=16)
#    plt.ylabel('Loss',fontsize=16)
#    plt.legend()
#    plt.figure()
#    plt.show()
#    fig1.savefig( log_folder + "/Training_X_Test_Loss.png", dpi=100)
#    
#    
#    plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
#    plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
#    fig2 = plt.gcf()
#    plt.title('Training and Test Accuracy')
#    plt.xlabel('Epochs ',fontsize=16)
#    plt.ylabel('Accuracy',fontsize=16)
#    plt.legend()
#    plt.figure()
#    plt.show()
#    fig2.savefig( log_folder + "/Training_X_Test_accuracy.png" , dpi=100)
#    
#    
#    with open(log_folder + "/log.txt", "w") as output:
#        print("###### Log 1 ###########\n\n" +
#              "Tran loss = " + str(train_loss) + "\n" + 
#              "Test Loss = " + str(test_loss) + "\n" + 
#              "Train Accuracy = " + str(train_accuracy) + "\n" +
#              "Test Accuracy = " + str(test_accuracy) + "\n\n" +
#              "With:\n" +
#              "epochs = " + str(epochs) + ",\n" +
#              "learning rate = " + str(learning_rate) + ", \n" +
#              "batch size = " + str(batch_size) + ", \n" +
#              "and with:\n" +
#              str(len(train)) + " Train Images and \n" +
#              str(len(test)) + " Test Images.\n\n" +
#              "########################\n" +
#              "END LOG", file=output)
#    
#    
#________________________ Running Iterations _________________________________#
#
#
#
#for i in range(0, n_iterations):
#    print("\n\n------ Iteration "+str(i)+"------\n\n")
#    multiple_iterations(e)
