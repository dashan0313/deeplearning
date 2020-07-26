import tensorflow as tf
import pathlib  #处理图片路径
import random
import os
import IPython.display as display
import matplotlib.pyplot as plt

# 程序选择最优的线程并行个数
AUTOTUNE = tf.data.experimental.AUTOTUNE
#预处理字符串形式的path
#data_root = pathlib.Path(data_root_orig)

def get_image_paths(data_root, keyword = '*/*',show_length = False):
    """
    data_root : path's name for root dir.
    keyword = '*/*', to search all the pictures in subdir.

    output:
    all_image_paths: Str
    """
    all_image_paths = list(data_root.glob(keyword))
    all_image_paths = [str(path) for path in all_image_paths]
    if show_length:
        print('The number of images:',len(all_image_paths))

    return all_image_paths

def show_pictures(all_image_paths, num = 3):
    for n in range(num):
        image_path = random.choice(all_image_paths)
        display.display(display.Image(image_path))
        print()

        return None

def get_label(data_root, keyword = '*/*',show_length = False):
    """
    outputs:
    label_names: list
    label_to_index: dict, name to [0-]
    all_image_labels: list
    """
    all_image_paths = get_image_paths(data_root,keyword,show_length)
    #is_dir()排除了TXT文件之类的，只保留目录类型文件
    #.name仅获取文件夹名字，不管前缀
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

    return label_names,label_to_index,all_image_labels

def preprocess_image(image,image_size = 192, channels = 3):

    """
    inputs:
    image: Str
    image_size: int
    channels: int, 1 or 3

    outputs:
    image: tensor, within 0-1, shape = (image_size, image_size, channels)

    """
    image = tf.image.decode_jpeg(image, channels)
    image = tf.image.resize(image, [image_size, image_size])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)


#创建tf.data.Dataset

def my_dataset(data_root, keyword = '*/*',show_length = False):

    all_image_paths = get_image_paths(data_root)
    label_names,label_to_index,all_image_labels = get_label(data_root)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    #image_count = len(all_image_paths)

    return image_label_ds
