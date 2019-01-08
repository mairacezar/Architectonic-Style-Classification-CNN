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
IMAGE_SIZE = 273

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



training_iters = 50
learning_rate = 0.001 
batch_size = 272

n_input = IMAGE_SIZE
n_classes = 3

tf.reset_default_graph()

x = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE,3])
y = tf.placeholder("float", [None, n_classes])


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,3,16), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,16,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*64,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(64,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(3), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_x)//batch_size):
            batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_x,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()






































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