import tensorflow as tf

from tensorflow.keras.applications import VGG16, ResNet50, MobileNet
from tensorflow.keras import layers
from tensorflow.keras.optimizers import *


def feature_extractor(params):
    """
    Loads the feature extraction part and set fine_tuning layers as trainable
    :param params:
    :return:
    """
    input_shape = (params['input_width'], params['input_depth'], 3)
    # load the feature extractor
    if params['feature_extractor']=='ResNet50':
        model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif params['feature_extractor']=='VGG16':
        model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif params['feature_extractor']=='MobileNet':
        model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    elif params['feature_extractor']=='MobileNet':
        model = MobileNet(weights='imagenet', include_top=False, input_shape=params['input_shape'])

    # check if fine tuning
    if params['fine_tuning']:
        # get the layer_name's number
        i = 0
        for layer in model.layers:
            if layer.name == params['layer_name']:
                break
            i += 1
        if i == 0:
            print("Layer name : ", params['layer_name'], ' Not found !')
        else:
            print(params['layer_name'], ' number = ', i)
        # Freeze all the layers after the name_layer
        for layer in model.layers[i + 1:]:
            layer.trainable = False
    else:
        # set all layer.trainable to False
        for layer in model.layers:
            layer.trainable = False
    return model


def classifier(feature_extractor, params):
    """
    set a head classifier on top of feature_extractor
    :param feature_extractor:
    :param params:
    :return:
    """
    # get number of output layer
    nbr_classes = len(params['classes'])
    model = tf.keras.models.Sequential()
    # Add the feature extractor convolutional base model
    model.add(feature_extractor)
    # flatten output
    model.add(layers.Flatten())
    # check mpl and add new layers
    if params['mlp']==1:
        model.add(layers.Dense(1024, activation='relu'))  # , kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(nbr_classes, activation='softmax'))
    elif params['mpl']==2:
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(nbr_classes, activation='softmax'))
    else:
        print('The classifier is not supported ')
    print(model.summary())

    # compile ...
    model.compile(optimizer=params['optimizer'],
                  loss=params['loss'],
                  metrics=[params['metrics']])
    return model
