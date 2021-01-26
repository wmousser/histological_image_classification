import os


def data_dict_initialization(params):
    """
    :param params:
    :return: data_dict = {'tumor_type': {'magnification': [] } }
    """
    data_dict = {}
    # initialize data_dict['tumor_type'] to contain magnifications
    for tumor_type in params['tumor_types']:
        data_dict[tumor_type] = {}
        for tumor_name in params[tumor_type]:
            data_dict[tumor_type][tumor_name] = {'40X': [], '100X': [], '200X': [], '400X': []}
    return data_dict


def get_magnification(root, params):
    """
    :param root: root as a string
    :return: the corresponding magnification as a string
    """
    for magnification in params['magnifications']:
        if magnification in root :
            return magnification
    # else return False
    return False


def get_tumor_name(root, params):
    """
    :param root: a path as string
    :return: the tumor type as a string
    """
    for tumor_type in params['tumor_types']:
        for tumor_name in params[tumor_type]:
            if tumor_name in root : return tumor_name

    return 'tumor name error'


def get_tumor_type(root, params):
    """

    :param root: as a path
    :param params:
    :return: tumor type from the benign or malignant
    """
    for tumor_type in params['tumor_types']:
        if tumor_type in root:
            return tumor_type
    # else False
    return 'tumor_type error'


def get_image_dict_from_folder(params):
    """
    :param dataset_path: dataset root folder
    :return: data_dict as {'tumor_type': {'magnification': [images_path] } }
    """

    # initialize data_dict as {'tumor_type': {'magnification': [] } }
    data_dict = data_dict_initialization(params)
    # Browse the dataset folder and construct the corresponding data dict
    for root, folders, files in os.walk(params['dataset_path']):
        if get_magnification(root, params) != False:
            tumor_type = get_tumor_type(root, params)
            tumor_name = get_tumor_name(root, params)
            magnification = get_magnification(root, params)
            for im_file in files:
                data_dict[tumor_type][tumor_name][magnification].append(root+'/'+im_file)
    return data_dict


def filter_dataset(data_dict ,params):
    """
    select data from data_dict where magnification and classes = {class_name : [image_list]}
    :param data_dict: {'tumor_type': {'magnification': [images_path] } }
    :param params:
    :return: dict({'class_name' : [image_list]})
    """
    filter_data = {}
    # initialize dict
    for class_name in params['classes']:
        if class_name in params['benign']:
            tumor_type = 'benign'
        elif class_name in params['malignant']:
            tumor_type = 'malignant'
        else : return False
        filter_data[class_name] = data_dict[tumor_type][class_name][params['magnification']]
    # show results
    print('Data filtered by %s magnification and the following classes :'%params['magnification'])
    for class_name in filter_data:
        print("%s : %i" %(class_name, len(filter_data[class_name])))
    return filter_data