"""
Create keras model
"""
import tensorflow as tf


def create_model(img_height, img_width, nb_classes, lr):
    """
    Creates keras model
    :param img_height: image height
    :param img_width: image width
    :param nb_classes: number of classes
    :param lr: optimizer learning rate
    :return: model
    """
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Convolution2D(
        input_shape=(img_height, img_width, 1),
        kernel_size=5,
        filters=32,
        padding='same',
        activation=tf.keras.activations.relu
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Convolution2D(
        kernel_size=3,
        filters=32,
        padding='same',
        activation=tf.keras.activations.relu,
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Convolution2D(
        kernel_size=3,
        filters=64,
        padding='same',
        activation=tf.keras.activations.relu
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=nb_classes, activation=tf.keras.activations.softmax))

    rms_prop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        optimizer=rms_prop_optimizer,
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    return model
