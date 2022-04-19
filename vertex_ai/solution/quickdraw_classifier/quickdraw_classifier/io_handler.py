"""
Read / Write functions
"""
import tensorflow as tf

from quickdraw_classifier.utils import parse_tfrecord


def get_img_tfrec_dataset(input_path: str, dataset_size: int, img_size: (int, int)):
    """
    Read a dataset from Google Cloud Storage and preprocess it
    :param input_path: input GCS path
    :param dataset_size: input dataset_size
    :param img_size: pair of image parameters (image height, image width)
    :return: shuffled dataset with (image, one_hot_class) parameters
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    AUTOTUNE = tf.data.AUTOTUNE

    filenames = tf.io.gfile.glob(input_path + "*.tfrec")

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(lambda img: parse_tfrecord(img, img_size), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda image, class_num, label, one_hot_class: (image, one_hot_class))
    dataset = dataset.shuffle(dataset_size)

    return dataset
