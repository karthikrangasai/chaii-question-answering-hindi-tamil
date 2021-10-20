import os

CHAII_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(CHAII_ROOT_PATH)

DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT_PATH, "data")

if not os.path.exists(os.path.join(PROJECT_ROOT_PATH, "my_logs")):
    os.mkdir(os.path.join(PROJECT_ROOT_PATH, "my_logs"))

OPTUNA_LOGS_PATH = os.path.join(PROJECT_ROOT_PATH, "my_logs", "optuna")
LR_RANGE_TEST_FIGS_PATH = os.path.join(PROJECT_ROOT_PATH, "my_logs", "lr_range_tests")

for path in [OPTUNA_LOGS_PATH, LR_RANGE_TEST_FIGS_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)

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
