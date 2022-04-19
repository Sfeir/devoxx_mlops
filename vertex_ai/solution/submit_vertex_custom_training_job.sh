REGION="europe-west1"
PROJECT_NAME="par-devoxx-sfeir"
SERVICE_ACCOUNT="sa-vertex@par-devoxx-sfeir.iam.gserviceaccount.com"

USERNAME="<TO DEFINE>"

DATE=$(date +"%Y%m%d_%H%M%S")
TRAINING_JOB_NAME="$USERNAME-quickdraw_training_$DATE"
MODEL_DISPLAY_NAME="$USERNAME-quickdraw_model_v01"

REPLICA_COUNT="1"
TRAINING_MACHINE_TYPE="n1-standard-8"
ACCELERATOR_TYPE="NVIDIA_TESLA_K80"
ACCELERATOR_COUNT="1"

TRAINING_EXECUTOR_IMAGE_URI="europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8:latest"
PREDICTION_EXECUTOR_IMAGE_URI="europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest"
PYTHON_MODULE="quickdraw_classifier.training"

GCS_TRAINING_DATA="gs://devoxx_quickdraw/tfrecord_data/training_data/"
GCS_VALIDATION_DATA="gs://devoxx_quickdraw/tfrecord_data/validation_data/"
GCS_MODEL_DATA_PATH="gs://$USERNAME-devoxx_quickdraw/gcs_model_data/quickdraw_classifier_$DATE/"
PACKAGE_URI="gs://$USERNAME-devoxx_quickdraw/vertex_job_code/quickdraw_classifier-0.0.1.tar.gz"


TRAINING_PIPELINE_REQUEST=\
"{
  'displayName': '$TRAINING_JOB_NAME',
  'trainingTaskDefinition': 'gs://google-cloud-aiplatform/schema/trainingjob/definition/custom_task_1.0.0.yaml',
  'trainingTaskInputs': {
    'serviceAccount': '$SERVICE_ACCOUNT',
    'baseOutputDirectory': {
      'outputUriPrefix': '$GCS_MODEL_DATA_PATH',
    },
    'workerPoolSpecs': [
      {
        'machineSpec': {
          'machineType': '$TRAINING_MACHINE_TYPE',
          'acceleratorType': '$ACCELERATOR_TYPE',
          'acceleratorCount': '$ACCELERATOR_COUNT'
        },
        'replicaCount': '$REPLICA_COUNT',
        'pythonPackageSpec': {
          'executorImageUri': '$TRAINING_EXECUTOR_IMAGE_URI',
          'packageUris': ['$PACKAGE_URI'],
          'pythonModule': '$PYTHON_MODULE',
          'args': [
            '--batch_size=50',
            '--validation_batch_size=20',
            '--training_ds_size=25000',
            '--validation_ds_size=5000',
            '--img_height=64',
            '--img_width=64',
            '--nb_classes=5'
          ],
          'env': [
            {
              'name': 'GCS_TRAINING_DATA',
              'value': '$GCS_TRAINING_DATA',
            },
            {
              'name': 'GCS_VALIDATION_DATA',
              'value': '$GCS_VALIDATION_DATA',
            },
            {
              'name': 'GCS_MODEL_DATA_PATH',
              'value': '$GCS_MODEL_DATA_PATH',
            }
          ]
        }
      }
    ],
  },
  'modelToUpload': {
    'displayName': '$MODEL_DISPLAY_NAME',
    'containerSpec': {
      'imageUri': '$PREDICTION_EXECUTOR_IMAGE_URI',
    },
  },
}"

echo $TRAINING_PIPELINE_REQUEST > training_request.json

curl -X POST \
-H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
-H "Content-Type: application/json; charset=utf-8" \
-d @training_request.json \
"https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_NAME/locations/$REGION/trainingPipelines"

