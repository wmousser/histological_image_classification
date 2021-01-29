import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt
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


def balance_dataset(data):
    """
    :param data:
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
    for set_name in ['valid', 'test']:  # , 'train']:
        print(set_name, 'is about ', len(data[set_name]['y']))
        for i in range(len(data[set_name]['x'])):
            image = cv2.imread(data[set_name]['x'][i])
            label = data[set_name]['y'][i]
            # image preprocessing
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

    # balance data
    data = balance_dataset(data)

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
