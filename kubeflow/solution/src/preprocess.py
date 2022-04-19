import math
import numpy as np
import tensorflow as tf
import datetime
import argparse

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Script to convert images > Tfrecords ')
# Paths must be passed in, not hardcoded
parser.add_argument('--input-path', type=str,
                    help='Path of the files containing the Input for training')
parser.add_argument('--output-path', type=str,
                    help='Path of the files where the Output training data should be written.')
parser.add_argument('--target-size', type=int,
                    help='Path of the files where the Output training data should be written.')
args = parser.parse_args()

print("Tensorflow version " + tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE

SHARDS = 5
CLASSES = [b'angel', b'cat', b'crown', b'the_eiffel_tower', b'the_mona_lisa']

GCS_TRAINING_PATTERN = args.input_path
GCS_TRAINING_TFRECORDS = args.output_path
TARGET_SIZE = args.target_size

print(GCS_TRAINING_PATTERN)
print(GCS_TRAINING_TFRECORDS)

nb_images = len(tf.io.gfile.glob(GCS_TRAINING_PATTERN))
shard_size = math.ceil(1.0 * nb_images / SHARDS)


def resize_and_crop_image(image, label):
    # Resize and crop using "fill" algorithm:
    # always make sure the resulting image
    # is cut out from the source image so that
    # it fills the TARGET_SIZE entirely with no
    # black bars and a preserved aspect ratio.
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(image, [w * tw / w, h * tw / w]),  # if true
                    lambda: tf.image.resize(image, [w * th / h, h * th / h])  # if false
                    )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label


# images are arranged in folders with corresponding labels
def decode_image_and_label(filename):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_png(bits)
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-2]
    return image, label


def recompress_image(image, label):
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_tfrecord(img_bytes, label):
    class_num = np.argmax(np.array(CLASSES) == label)
    one_hot_class = np.eye(len(CLASSES))[class_num]

    feature = {
        "image": _bytes_feature([img_bytes]),
        "class_num": _int64_feature([class_num]),
        "label": _bytes_feature([label]),
        "one_hot_class": _float_feature(one_hot_class.tolist())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


filenames = tf.data.Dataset.list_files(GCS_TRAINING_PATTERN, seed=35155)  # This also shuffles the images
quickdraw_dataset = filenames.map(decode_image_and_label, num_parallel_calls=AUTOTUNE)
quickdraw_dataset = quickdraw_dataset.map(resize_and_crop_image, num_parallel_calls=AUTOTUNE)
quickdraw_dataset = quickdraw_dataset.map(recompress_image, num_parallel_calls=AUTOTUNE)

# sharding: there will be one "batch" of images per file 
quickdraw_dataset = quickdraw_dataset.batch(shard_size)


def write_dataset(dataset, filepath):
    print(f'Starting writing {datetime.datetime.now().strftime("%H:%M:%S.%f")}')
    dataset = dataset.enumerate()
    for shard, (image, label) in dataset:
        shard_size = image.numpy().shape[0]
        filename = filepath + f"quickdraw_dataset{str(shard.numpy()).rjust(2, '0')}_{shard_size}.tfrec"
        print(f'Starting file writing {datetime.datetime.now().strftime("%H:%M:%S.%f")}')

        with tf.io.TFRecordWriter(filename) as tf_writer:
            for i in range(shard_size):
                example = to_tfrecord(image.numpy()[i], label.numpy()[i])
                tf_writer.write(example.SerializeToString())
            print(f'Wrote file {filename} containing {shard_size} records')
            print(f'Wrote file at {datetime.datetime.now().strftime("%H:%M:%S.%f")}')


write_dataset(quickdraw_dataset, GCS_TRAINING_TFRECORDS)
