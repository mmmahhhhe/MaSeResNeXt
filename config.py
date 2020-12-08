import codecs
import os
import utils

train_parameters = {  
    "data_dir": "dataset",  # data set directory
    "infer_img": 'dataset/evalImageSet/metamorphic_4_Andalusite_hornfels_1-6.jpg',  # infer img path
    "num_epochs": 1000,  
    "train_batch_size": 64, 
    "mean_rgb": [85, 96, 102],  # RGB
    "input_size": [3, 224, 224],  
    "class_dim": -1,  # number of categories
    "image_count": -1,  # number of training img
    "label_dict": {},  
    "train_file_list": "train.txt",  # training set filename path
    "eval_file_list": "eval.txt",  
    "label_file": "label_list.txt",  # the mapping with name and label
    "save_model_dir": "./save_dir/model",  # model saving path
    "continue_train": True,      # whether continue to training
    "image_enhance_strategy": {  # Image enhancement correlation strategy 
        "need_distort": True,    # Whether to enable image color enhancement
        "need_rotate": True,     # Whether to increase the random Angle
        "need_crop": True,       # Whether to add clipping
        "need_flip": True,       # Whether to add horizontal random flips or not
        "hue_prob": 0.5,  
        "hue_delta": 18,  
        "contrast_prob": 0.5,  
        "contrast_delta": 0.5,  
        "saturation_prob": 0.5,  
        "saturation_delta": 0.5,  
        "brightness_prob": 0.5,  
        "brightness_delta": 0.125  
    },  
    "early_stop": {  
        "sample_frequency": 50,  
        "successive_limit": 3,  
        "good_acc1": 0.92  
    },  
    "rsm_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "momentum_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "sgd_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "adam_strategy": {  
        "learning_rate": 0.002  
    }  
}  

def init_train_parameters():
    """
    Initialize training parameters
    :return:
    """

    label_list = os.path.join(train_parameters['data_dir'], train_parameters['label_file'])
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            parts = line.strip().split()
            train_parameters['label_dict'][parts[1]] = int(parts[0])
            index += 1
        train_parameters['class_dim'] = index

    train_file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_file_list'])
    with codecs.open(train_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['train_image_count'] = len(lines)

    eval_file_list = os.path.join(train_parameters['data_dir'], train_parameters['eval_file_list'])
    with codecs.open(eval_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['eval_image_count'] = len(lines)

    utils.logger.info("input_size: {}".format(train_parameters['input_size']))
    utils.logger.info("class_dim: {}".format(train_parameters['class_dim']))
    utils.logger.info("continue_train: {}".format(train_parameters['continue_train']))
    utils.logger.info("train_image_count: {}".format(train_parameters['train_image_count']))
    utils.logger.info("eval_image_count: {}".format(train_parameters['eval_image_count']))
    utils.logger.info("num_epochs: {}".format(train_parameters['num_epochs']))
    utils.logger.info("train_batch_size: {}".format(train_parameters['train_batch_size']))
    utils.logger.info("mean_rgb: {}".format(train_parameters['mean_rgb']))
    utils.logger.info("save_model_dir: {}".format(train_parameters['save_model_dir']))


init_train_parameters()