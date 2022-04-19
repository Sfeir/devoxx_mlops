"""
Main training
"""
import logging
import os

import click

import tensorflow as tf
import hypertune

from quickdraw_classifier import (GCS_TRAINING_DATA, GCS_VALIDATION_DATA, GCS_MODEL_DATA_PATH)
from quickdraw_classifier.io_handler import get_img_tfrec_dataset
from quickdraw_classifier.model import create_model


@click.command()
@click.option("--batch_size", required=True, type=int)
@click.option("--validation_batch_size", required=True, type=int)
@click.option("--training_ds_size", required=True, type=int)
@click.option("--validation_ds_size", required=True, type=int)
@click.option("--img_height", required=True, type=int)
@click.option("--img_width", required=True, type=int)
@click.option("--nb_classes", required=True, type=int)
@click.option("--lr", required=False, type=float, default=0.001)
@click.option("--num_epochs", required=False, type=int, default=5)
def main(batch_size: int, validation_batch_size: int, training_ds_size: int, validation_ds_size: int,
         img_height: int, img_width: int, nb_classes: int, lr: float, num_epochs: int):
    strategy = tf.distribute.get_strategy()

    logging.info(f"Reading training dataset at {GCS_TRAINING_DATA}")
    training_dataset = get_img_tfrec_dataset(GCS_TRAINING_DATA, training_ds_size, (img_height, img_width))
    training_dataset = training_dataset.batch(batch_size)

    logging.info(f"Reading validation dataset at {GCS_VALIDATION_DATA}")
    validation_dataset = get_img_tfrec_dataset(GCS_VALIDATION_DATA, validation_ds_size, (img_height, img_width))
    validation_dataset = validation_dataset.batch(validation_batch_size)

    logging.info(f"Defining checkpoint and early stopping callbacks")
    gcs_checkpoint_path = os.path.join(GCS_MODEL_DATA_PATH, "model", "")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(gcs_checkpoint_path, save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=2, restore_best_weights=True
    )
    with strategy.scope():
        model = create_model(img_height, img_width, nb_classes, lr)

    logging.info("Starting model fitting ...")
    history = model.fit(
        training_dataset,
        epochs=num_epochs,
        validation_data=validation_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1
    )

    hp_metric = history.history['val_accuracy'][-1]

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='val_accuracy',
                                            metric_value=hp_metric,
                                            global_step=num_epochs)


if __name__ == "__main__":
    main()
