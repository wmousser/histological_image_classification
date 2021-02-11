import cv2
import numpy as np
import datetime
import collections
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import random_zoom, random_rotation, load_img, save_img, img_to_array
from utils.utils_breakHis import *
from utils.models import *
from sklearn.preprocessing import LabelEncoder
from random import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def print_dict_values(name, dictionary):
    """
    :param name:
    :param dictionary:
    :return: print keys and values of dictionary
    """
    print('-' * 100, '\nThe dict |  ', name, '  |   contains :\n', '-' * 100)
    for el in dictionary:
        print(el, dictionary[el])
    print(100 * '-', '\n', 100 * '-')


def get_split_values(nbr_items, test_split, valid_split):
    """
    :param nbr_items: nbr images to split
    :param test_split:
    :param valid_split:
    :return: nbr items for train test and valid sets
    """
    nbr_test = int(nbr_items * test_split)
    nbr_valid = int((nbr_items - nbr_test) * valid_split)
    nbr_train = nbr_items - nbr_test - nbr_valid
    # show values
    # print('%d for train, %d for valid and %d for test'%(nbr_train,nbr_valid,nbr_test))
    return nbr_train, nbr_valid, nbr_test


def train_test_split_data_dict(data_dict, params):
    """
    split into train test and valid
    :param data_dict:
    :param params:
    :return: dict({'train': {'x':[image_path], 'y':[label_list]},
                    'valid': {{'x':[image_path], 'y':[label_list]},
                    'test': {{'x':[image_path], 'y':[label_list]}
                 })
    """
    data_split = initialize_data()
    test_split = params['test_split']
    valid_split = params['valid_split']
    for class_name in data_dict:
        # work on an image list
        image_list = data_dict[class_name]
        # shuffle image list
        shuffle(image_list)
        # get valid, train and test values
        nbr_train, nbr_valid, nbr_test = get_split_values(nbr_items=len(image_list),
                                                          test_split=test_split,
                                                          valid_split=valid_split)

        data_split['train']['x'].extend(image_list[:nbr_train])
        data_split['train']['y'].extend([class_name] * nbr_train)

        valid_list = image_list[nbr_train:nbr_train + nbr_valid]

        data_split['valid']['x'].extend(valid_list)
        data_split['valid']['y'].extend([class_name] * nbr_valid)

        data_split['test']['x'].extend(image_list[nbr_train + nbr_valid:])
        data_split['test']['y'].extend([class_name] * nbr_test)
    # resume results
    print('Split : %d images for train, %d images for valid and %d images for test'
          % (len(data_split['train']['x']),
             len(data_split['valid']['x']),
             len(data_split['test']['x'])))

    return data_split


def initialize_data():
    """
    :return: an empty data dict
    """
    return {'train': {'x': [], 'y': []},
            'valid': {'x': [], 'y': []},
            'test': {'x': [], 'y': []}
            }


def from_dict_path_to_data_dict(data):
    """
    :param data:
    :return: image as numpy array data_dict
    """
    # initialize the resulting data_dict
    image_data = initialize_data()
    # todo: convert images from test train and valid sets
    for set_name in ['valid']:  #, 'test' , 'train']:
        print(set_name, 'is about ', len(data[set_name]['y']))
        # loop for each image
        for i in range(len(data[set_name]['x'])):
            # preprocess image
            image = preprocess_image(data[set_name]['x'][i])
            label = data[set_name]['y'][i]
            # update the data_dict
            image_data[set_name] = append_im_data_dict(image_data[set_name], image, label)
    return image_data


def compute_nbr_images_to_add(nbr_images, target_nbr):
    """
    :param total:
    :param target_nbr:
    :return: nbr of images used to balance data
    """
    return int((target_nbr - nbr_images)/8)


def balance_dataset(data, params):
    """
    :param data:
    :return: preprocess then balance data applying rotations and zoom
    """
    balanced_data = initialize_data()
    # temp dict to balance images per class
    temps_dict = {}
    # initialize content
    print('-'*30,'   Balancing   ','-'*30)
    for cl in params['classes']:
        temps_dict[cl] = []
    # todo : loop over train, test and valid sets
    for set_name in ['valid']:#, 'test' , 'train']:
        print(set_name, 'is about ', len(data[set_name]['y']))
        # get the nbr of images to have
        target_nbr = max(collections.Counter(data[set_name]['y']).values())
        # fill in the temp_dict
        for cl in params['classes']:
            for index, img in enumerate(data[set_name]['x']):
                if data[set_name]['y'][index] == cl:
                    temps_dict[cl].append(data[set_name]['x'][index])
        # balance class per class
        for label in params['classes']: #d
            # number of available images
            nbr_images = len(temps_dict[cl])
            # number of images to use for balancing
            images_to_add = compute_nbr_images_to_add(nbr_images, target_nbr)
            # create new images and append them into the balanced_data
            for i in range(images_to_add):
                # working on image
                image = temps_dict[cl][i]
                # insert images into balanced_data
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], image, label)

                # ROTATE_75_CLOCKWISE
                im = random_rotation(image, rg=75, fill_mode='reflect')
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)
                # ROTATE_75_ANTICLOCKWISE
                im = random_rotation(image, rg=-75, fill_mode='reflect')
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)

                # ROTATE_15_CLOCKWISE
                im = random_rotation(image, rg=15, fill_mode='reflect')
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)
                # ROTATE_15_ANTICLOCKWISE
                im = random_rotation(image, rg=-15, fill_mode='reflect')
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)

                # Zoom 0.65, 0.75, 0.8, 0.9
                im = random_zoom(image, zoom_range=(0.65, 0.65), fill_mode='reflect')
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)
                im = random_zoom(image, zoom_range=(0.75, 0.75), fill_mode='reflect')
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)
                im = random_zoom(image, zoom_range=(0.8, 0.8), fill_mode='reflect')
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)
                im = random_zoom(image, zoom_range=(0.9, 0.9), fill_mode='reflect')
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)
            # add / insert the remaining images without rotations
            for j in range(i,nbr_images):
                im = temps_dict[cl][j]
                balanced_data[set_name] = append_im_data_dict(balanced_data[set_name], im, label)

        # Show resume
        print(set_name, 'balanced with ', len(balanced_data[set_name]['y']))
    # return balanced data
    return balanced_data


def append_im_data_dict(data, im, label):
    """
    :param data: dict 'x': [..] , ['y']:[..]
    :param im:
    :param label:
    :return: append image into x and label into y
    """
    data['x'].append(im)
    data['y'].append(label)

    return data


def preprocess_image(image_path):
    """
    Loads image from image_path
    :param image_path:
    :return: a rescaled image [0, 1]
    """

    # image preprocessing
    image = load_img(image_path, color_mode='rgb')
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255
    return image


def augment_data(data, params):
    """
    :param data:
    :param params:
    :return: augment data using flip and rotations
    """
    # set an dict
    augmented_data = initialize_data()
    print('-'*30,'   Augmentations   ','-'*30)
    # todo: loop over test, train and validation sets
    for set_name in ['valid']: #, 'test', 'train']:
        print(set_name, 'is about ', len(data[set_name]['y']))
        for i in range(len(data[set_name]['x'])):
            image = data[set_name]['x'][i]
            label = data[set_name]['y'][i]
            # resize and insert the current image
            image = cv2.resize(image, (params['input_width'], params['input_depth']))
            augmented_data[set_name] = append_im_data_dict(augmented_data[set_name], image, label)
            # ROTATE_90_CLOCKWISE
            im = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
            augmented_data[set_name] = append_im_data_dict(augmented_data[set_name], im, label)
            # ROTATE_90_COUNTERCLOCKWISE
            im = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            augmented_data[set_name] = append_im_data_dict(augmented_data[set_name], im, label)
            # get flip params
            for flip in params['flip']:
                im = cv2.flip(image, flip)
                augmented_data[set_name] = append_im_data_dict(augmented_data[set_name], im, label)
                if flip == 0:
                    im = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
                    im = cv2.flip(im, flip)
                    augmented_data[set_name] = append_im_data_dict(augmented_data[set_name], im, label)

                    im = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                    im = cv2.flip(im, flip)
                    augmented_data[set_name] = append_im_data_dict(augmented_data[set_name], im, label)
        print(set_name, 'augmented by ', len(augmented_data[set_name]['y']))
    return augmented_data


def shuffle_dataset(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    if not x:
        return x, y
    # zip sets
    temp_zip = list(zip(x, y))
    # shuffle
    shuffle(temp_zip)
    # unzip
    x, y = zip(*temp_zip)
    # convert resulting sets into lists
    return list(x), list(y)


def to_categorical(y_set, class_list):
    """
    :param y_set: int labels
    :param class_list:
    :return: binary class matrix
    """
    encoder = LabelEncoder()
    y_enc = encoder.fit(y_set)
    y_set = y_enc.transform(y_set)
    y_set = tf.keras.utils.to_categorical(y=y_set, num_classes=len(class_list))

    return y_set


def build_datasets(data, params):
    """
    :param params: 
    :param data:
    :return: shuffled datasets x_train, y_train, x_test, y_test, x_valid, y_valid
    """
    x_train = data['train']['x']
    y_train = data['train']['y']
    # shuffle
    x_train, y_train = shuffle_dataset(x_train, y_train)

    x_test = data['test']['x']
    y_test = data['test']['y']
    # shuffle
    x_test, y_test = shuffle_dataset(x_test, y_test)

    x_valid = data['valid']['x']
    y_valid = data['valid']['y']
    # shuffle
    x_valid, y_valid = shuffle_dataset(x_valid, y_valid)

    # as array
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    x_valid = np.asarray(x_valid)

    # convert label vectors
    y_train = np.asarray(to_categorical(y_train, params['classes']))
    y_valid = np.asarray(to_categorical(y_valid, params['classes']))

    return (x_train, y_train,
            x_test, y_test,
            x_valid, y_valid)


def prepare_data(params):
    """
    :param params:
    :return: x_train, y_train, x_test, y_test, x_valid, y_valid
    """
    # prepare a data dict from a directory
    data = get_image_dict_from_folder(params)

    # filter dataset to use (magnification, classes ..)
    data = filter_dataset(data, params)

    # split data into train, test and validation sets
    data = train_test_split_data_dict(data, params)

    # load images as numpy array
    data = from_dict_path_to_data_dict(data)

    # balance data
    data = balance_dataset(data, params)

    # augmentations
    data = augment_data(data, params)

    # build datasets
    x_train, y_train, x_test, y_test, x_valid, y_valid = build_datasets(data, params)

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def prepare_model(x_train, y_train, x_valid, y_valid, params):
    """
    :param x_valid: 
    :param x_train: 
    :param y_valid: 
    :param params:
    :return: compiled model
    """
    # set feature extraction part
    model = feature_extractor(params)

    # add classification part
    model = classifier(model, params)

    # callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=params['early_stopping_patience'], verbose=2)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(params['model_name'], save_best_only=True, verbose=2)
    log_dir = params['output_dir'] + params['model_name'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    params['log_dir'] = log_dir + '/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # todo : comment the following
    # (x_train, y_train) = (x_valid, y_valid)

    # fit(model, data)
    model.fit(x=x_train, y=y_train,
              batch_size=params['train_batch_size'],
              epochs=params['epochs'],
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, tensorboard_callback],
              validation_data=(x_valid, y_valid),
              )

    return model


def perform_predictions(model, x_test, y_test, params):
    """
    :param model:
    :param x_test:
    :param y_test:
    :param params:
    :return: print classification reports
    """
    # test the classifier
    y_pred = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)

    y_test = np.asarray(y_test)
    # encode y_test set
    encoder = LabelEncoder()
    encoder.fit(y_test)

    # decode y_enc set
    y_pred_labels = encoder.inverse_transform(y_pred)

    # get y_test label list
    labels = list(set(y_test))
    # confusion matrix reports and plot
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_labels, labels=labels)
    # classifications reports
    print(classification_report(y_test, y_pred_labels))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(include_values=True, cmap='plasma', xticks_rotation='-45',
              values_format=None, ax=None)
    plt.title(params['model_name'])
    plt.savefig(params['log_dir'] + params['model_name'])
