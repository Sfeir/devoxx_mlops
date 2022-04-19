""" Env Initialization"""
from os import environ

GCS_TRAINING_DATA = environ["GCS_TRAINING_DATA"]
GCS_VALIDATION_DATA = environ["GCS_VALIDATION_DATA"]
GCS_MODEL_DATA_PATH = environ["GCS_MODEL_DATA_PATH"]

# if the job is used only for training, it's possible to use AIP_MODEL_DIR instead of GCS_MODEL_DATA_PATH
# it's an environment variable directly set by Vertex AI
# AIP_MODEL_DIR: a Cloud Storage URI of a directory intended for saving model artifacts
# MODEL_DIR = environ["AIP_MODEL_DIR"]
