REGION="europe-west1"
PROJECT_NAME="par-devoxx-sfeir"
SERVICE_ACCOUNT="sa-vertex@par-devoxx-sfeir.iam.gserviceaccount.com"

USERNAME="<TO DEFINE>"

DATE=$(date +"%Y%m%d_%H%M%S")
TRAINING_JOB_NAME="$USERNAME-quickdraw_hp_tunning_$DATE"

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


HP_TUNING_REQUEST=\
"{
  'displayName': '$TRAINING_JOB_NAME',
  'studySpec': {
    'metrics': [
      {
        'metricId': 'val_accuracy',
        'goal': 'MAXIMIZE'
      }
    ],
    'parameters': [
      {
        'parameterId': 'lr',
        'doubleValueSpec': {
          'minValue': 1e-03,
          'maxValue': 5e-03
        },
        'scaleType':'UNIT_LINEAR_SCALE'
      },
      {
        'parameterId': 'batch_size',
        'discreteValueSpec': {
            'values': [
              50, 100, 150
            ]
        }
      }
    ]
  },
  'maxTrialCount': 8,
  'parallelTrialCount': 2,
  'maxFailedTrialCount': 3,
  'trialJobSpec': {
    'serviceAccount': '$SERVICE_ACCOUNT',
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
  }
}"

echo $HP_TUNING_REQUEST > hp_tuning_request.json

curl -X POST \
-H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
-H "Content-Type: application/json; charset=utf-8" \
-d @hp_tuning_request.json \
"https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_NAME/locations/$REGION/hyperparameterTuningJobs"

