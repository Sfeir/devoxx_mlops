"""
Helper functions
"""
import tensorflow as tf


def parse_tfrecord(example, img_size):
    """
    Parse tf record that represents an image of size img_size,
    and contains additional information such as : class number, label and one hot encoded class
    :param example: TFRecord encoded image
    :param img_size: image size
    :return:
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class_num": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.string),
        "one_hot_class": tf.io.VarLenFeature(tf.float32)
    }

    example = tf.io.parse_single_example(example, features)

    image = tf.io.decode_jpeg(example['image'], channels=1)
    image = tf.reshape(image, [*img_size, 1])

    class_num = example['class_num']
    label = example['label']
    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])

    return image, class_num, label, one_hot_class
