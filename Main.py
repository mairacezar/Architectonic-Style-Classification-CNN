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
n_iterations = int( input("Numeber of iterations: ") )
e = int( input("Number of epochs: ") )

def multiple_iterations(n_epochs):
    

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
                np.fliplr(aug_data[i][0])

            case = randint(0,1)    
            
            if (case == 1):
                random_noise(aug_data[i][0])

            case = randint(0,1)
            
            #if (case == 1):
#                print("exposure")
#                plt.imshow(aug_data[i][0])
#                plt.show()
                
                #v_min, v_max = np.percentile(aug_data[i][0], (0.2, 99.8))
                #aug_data[i][0] = exposure.rescale_intensity(aug_data[i][0], in_range=(v_min, v_max))
                
#                plt.imshow(aug_data[i][0])
#                plt.show()
        return aug_data
    new_data = augmentation(data)
    print(np.shape(new_data))
    print(new_data[1][0].shape)
    #np.append(data, augmentation(data), 0)
    
#    plt.imshow(data[0][0])    
#    plt.show()
#    plt.imshow(data[387][0])
#    plt.show()
    
    #shuffle(data)
    
#    print(data[0][0].shape)
#    print(data[387][0].shape)


    # ENG: Divides dataset into Train and test ( 272 and 30 data each)
    # PT: Divide o dataset em Train e Test (272 e 30 dados cada)
    train = data[:-59]
    test = data[-59:]
    
    
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
    plt.imshow(train_x[0])
    plt.show()
    
    # Print image label for sanity checks
    print(train_y[0], train_y.shape)
    
    
#____________________CONSTRUCT CONVOLUTIONAL NEURAL NETWORK___________________#
    
    tf.reset_default_graph()
    
    #hyperparameters
    epochs = n_epochs
    learning_rate = 1e-5
    batch_size = 164
    classes = NUM_CLASSES
    input_shape = IMAGE_SIZE
    kernel_size = 3
    
    #placehouder to recive input and output
    image_input = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_output = tf.placeholder("float", [None, classes])
    
    #convolutional layer function for apply kernels
    def conv2d(x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    #max pooling function for backpropagation
    def max_pooling(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    #calculate wd1 size
    def calc_wd1(kernel_size, input_shape):    
        final_size = input_shape
        for i in range(0,3):
            size = int(int(final_size+1)/int(2)) ###########era pra ser -3 que é o tamanho do filtro, mas -1 tá funcionando p todos os casos       
            final_size = size
            print(final_size)
        return final_size
    
    wd1_size = calc_wd1(kernel_size, input_shape)
    
        
    #defining bias and weights dictionary for set then in tensorflow cnn functions
    weights = {
            'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer = tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer = tf.contrib.layers.xavier_initializer()),
            'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer = tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable('W3', shape=(wd1_size*wd1_size*128,128), initializer = tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W6', shape=(128,classes), initializer = tf.contrib.layers.xavier_initializer()),
            }
    
    biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer = tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer = tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(128), initializer = tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(128), initializer = tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B4', shape=(3), initializer = tf.contrib.layers.xavier_initializer()),
            }
    
    def convolutional_net(x, weights, bias):
        
        #neural network layers
        conv1 = conv2d(x, weights['wc1'], biases['bc1'] )
        conv1 = max_pooling(conv1, k=2)
    #    print(conv1.shape)
        
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = max_pooling(conv2, k=2)
        
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = max_pooling(conv3, k=2)
    #    print(conv2.shape)
    #    print(conv3.shape)
        
        #fully connected layer with activation function
        fc1 = tf.reshape(conv3, [-1,weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        
        #output
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        
        return out
    
    #model
    prediction = convolutional_net(image_input, weights, biases)
    
    #loss, activiation and backpropagation functions for model
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=label_output))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    #test perform
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #variables initialization
    init = tf.global_variables_initializer()
    
    #initiate train
    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        summary = tf.summary.FileWriter('./Output', sess.graph)
        
        for i in range(epochs):
            for batch in range(len(train_x)//batch_size):
                batch_x = train_x[batch*batch_size:min((batch+1*batch_size, len(train_x)))]
                batch_y = train_y[batch*batch_size:min((batch+1*batch_size, len(train_y)))]
                
                #TODO: Essa variável não é usada
                back = sess.run(optimizer, feed_dict = {image_input: batch_x, label_output: batch_y} )
        
                loss, acc = sess.run([cost, accuracy], feed_dict = {image_input: batch_x, label_output: batch_y} )
                
            print("Epoch " + str(i) + ", Loss = " + \
                          "{:.6f}".format(loss) + ", Training Accuracy = " + \
                          "{:.5f}".format(acc))
            print("Backpropagation complete.")
            
            
            test_acc, valid_loss = sess.run([accuracy, cost], feed_dict = {image_input: test_x, label_output: test_y})
            train_loss.append(loss)
            test_loss.append(valid_loss)
            train_accuracy.append(acc)
            test_accuracy.append(test_acc)
            print("Testing Accuracy:","{:.5f}".format(test_acc))
            print("Valid Loss","{:.5f}".format(valid_loss))
            
        summary.close()
        
#_______________________________ Logging _____________________________________#
    
    
    import datetime
    now = datetime.datetime.now()
    
    time = now.strftime("Log %d-%m-%Y - %H-%M-%S")
    
    os.makedirs( os.path.join( log_save_directory, time ) )
    
    log_folder = os.path.join(log_save_directory, time)
    
    
    
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
    
    #axes1 = plt.gca()
    #axes1.set_ylim([0,0.1])
    
    fig1 = plt.gcf()
    plt.title('Training and Test loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
    fig1.savefig( log_folder + "/Training_X_Test_Loss.png", dpi=100)
    
    
    plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
    fig2 = plt.gcf()
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
    fig2.savefig( log_folder + "/Training_X_Test_accuracy.png" , dpi=100)
    
    
    with open(log_folder + "/log.txt", "w") as output:
        print("###### Log 1 ###########\n\n" +
              "Tran loss = " + str(train_loss) + "\n" + 
              "Test Loss = " + str(test_loss) + "\n" + 
              "Train Accuracy = " + str(train_accuracy) + "\n" +
              "Test Accuracy = " + str(test_accuracy) + "\n\n" +
              "With:\n" +
              "epochs = " + str(epochs) + ",\n" +
              "learning rate = " + str(learning_rate) + ", \n" +
              "batch size = " + str(batch_size) + ", \n" +
              "and with:\n" +
              str(len(train)) + " Train Images and \n" +
              str(len(test)) + " Test Images.\n\n" +
              "########################\n" +
              "END LOG", file=output)
    
    
#________________________ Running Iterations _________________________________#



for i in range(0, n_iterations):
    print("\n\n------ Iteration "+str(i)+"------\n\n")
    multiple_iterations(e)
