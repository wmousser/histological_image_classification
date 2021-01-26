import pandas
import  cv2
import collections
import numpy as np
from PIL import Image
from utils.utils_breakHis import *
from random import shuffle

def print_dict_values(name, dictionary):
    '''
    :param dictionary:
    :return: print keys and values of dictionary
    '''
    print('-'*100, '\nThe dict |  ', name, '  |   contains :\n','-'*100)
    for el in dictionary:
        print(el , dictionary[el])
    print(100*'-','\n', 100*'-')


def get_split_values(nbr_items, test_split, valid_split):
    """
    :param nbr_items: nbr images to split
    :param test_split:
    :param valid_split:
    :return: nbr items for train test and valid sets
    """
    nbr_test = int(nbr_items * test_split)
    nbr_valid = int((nbr_items - nbr_test)* valid_split)
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
        data_split['train']['y'].extend([class_name]*nbr_train)

        valid_list = image_list[nbr_train:nbr_train+nbr_valid]

        data_split['valid']['x'].extend(valid_list)
        data_split['valid']['y'].extend([class_name] * nbr_valid)

        data_split['test']['x'].extend(image_list[nbr_train+nbr_valid:])
        data_split['test']['y'].extend([class_name] * nbr_test)
    #resume results
    print('Split : %d images for train, %d images for valid and %d images for test'
          %(len(data_split['train']['x']),
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


def balance_dataset(data):
    """
    :param df:
    :param set_name:
    :return: todo : balanced dataset ...
    """

    # balanced data
    data_balanced = initialize_data()
    return data


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


def augment_data(data, params):
    """
    :param data:
    :param params:
    :return: Preprocess then augment data using flip and rotations
    """
    # set an dict
    augmented_data = initialize_data()
    # load, convert and preprocess images
    for set_name in ['valid'] : #, 'train', 'test']:
        print(set_name, 'is about ', len(data[set_name]['y']))
        for i in range(len(data[set_name]['x'])):
            image = cv2.imread(data[set_name]['x'][i])
            label = data[set_name]['y'][i]
            image = np.asarray(image)
            image = image.astype('float32')
            image /= 255
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
        print(set_name , 'augmented by ',len(augmented_data[set_name]['y']))
    return augmented_data


def build_datasets(data):
    x_train = data['train']['x']
    y_train = data['train']['y']

    x_test = data['test']['x']
    y_test = data['test']['y']

    x_valid = data['valid']['x']
    y_valid = data['valid']['y']

    return (x_train, y_train,
            x_test, y_test,
            x_valid, y_valid)


def prepare_data(params):
    # from a directory prepare a data dict
    data = get_image_dict_from_folder(params)

    # filter data to use (magnification, classes ..)
    data = filter_dataset(data, params)

    # split data into train, test and validation sets
    data = train_test_split_data_dict(data, params)

    # balance data
    data = balance_dataset(data)

    # augmentations
    data = augment_data(data, params)

    # build datasets
    x_train, y_train, x_test, y_test, x_valid, y_valid = build_datasets(data)


    # flow from datframe

