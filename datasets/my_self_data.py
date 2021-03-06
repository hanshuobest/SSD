'''
Provides data for custom data
author:hanshuo
'''

import tensorflow as tf
from datasets import pascalvoc_common
slim = tf.contrib.slim

FILE_PATTERN = 'my_self_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (670, 865),
    'bicycle': (552, 711),
    'bird': (765, 1119),
    'boat': (508, 850),
    'bottle': (706, 1259),
    'bus': (421, 593),
    'car': (1161, 2017),
    'cat': (1080, 1217),
    'chair': (1119, 2354),
    'cow': (303, 588),
    'diningtable': (538, 609),
    'dog': (1286, 1515),
    'horse': (482, 710),
    'motorbike': (526, 713),
    'person': (4087, 8566),
    'pottedplant': (527, 973),
    'sheep': (325, 813),
    'sofa': (507, 566),
    'train': (544, 628),
    'tvmonitor': (575, 784),
    'total': (11540, 27450),
}
SPLITS_TO_SIZES = {
    'train': 17125,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
}
NUM_CLASSES = 20
def get_split(split_name , dataset_dir , file_pattern = None , reader = None):
    '''

    :param split_name:
    :param dataset_dir:
    :param file_pattern:
    :param reader:
    :return:
    '''
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return pascalvoc_common.get_split(split_name , dataset_dir , file_pattern , reader , SPLITS_TO_SIZES , ITEMS_TO_DESCRIPTIONS , NUM_CLASSES)
