import os

CHAII_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(CHAII_ROOT_PATH)

DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT_PATH, "data")

MODEL_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT_PATH, "model_checkpoints")

if not os.path.exists(MODEL_CHECKPOINT_PATH):
    os.mkdir(MODEL_CHECKPOINT_PATH)

TODO = [
    "Combine Datasets",
    "Preprocess Dataset",
    "Test Data Pipeline",
    "Test Model Creation",
    "Test Trainer",
    "Add Finetuning Callbacks",
    "Create configurations - pretrain, finetune",
    "Create configurations - sweeps",
    "Create Sweep Script",
    "Create Training scripts",
]
