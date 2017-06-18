from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.preprocessing import image
from tensorflow.contrib.keras.python.keras.preprocessing.image import img_to_array, load_img
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from matplotlib import pylab as plt


#modified from Jeremy Howard's VGG16BN

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical', seed = None):
    """Retrieve images in batches

    #Arguments
     path: path to images

    #Returns
     generator creating data batches
    """
    return gen.flow_from_directory(path, target_size=(224, 224),
            class_mode = class_mode, shuffle = shuffle, batch_size = batch_size, seed = seed)


def finetune(base_model, n_class, lr = 1e-3):
    """Refine a given model by removing the last layer and adding a custom new one

    #Arguments
     base_model: model to start from
     n_class: number of classes

    #Returns
     Model where only the last layer is trained
    """

    for layer in base_model.layers: layer.trainable = False

    x  = base_model.layers[-2].output
    x = Dense(n_class, activation = 'softmax')(x)

    model = Model(inputs = base_model.input, outputs = x)
    model.compile(optimizer = Adam(lr = lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


def show_image(path):
    """Print an image to the notebook

    #Arguments
     path: location of image
    """
    show_array(img_to_array(load_img(path)))
    
def show_array(arr, figure = None, cmap = None, figsize = (12, 12)):
    """Print an image given as array to the notebook

    #Arguments
     arr: array-image to be printed
    """

    arr= arr[0] if len(arr.shape) == 4 else arr
    if figure: 
        figure.imshow(arr/255)
    else:
        plt.figure(figsize = figsize)
        plt.imshow(arr/255, cmap = cmap)

def show_array_list(arr_list):
    """Print a list of images as array to the notebook

    #Arguments
     arr_list: list of arrays
    """
    fig, axs = plt.subplots(1, len(arr_list))
    for arr,ax in zip(arr_list,axs): show_array(arr, ax)
   
