import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import flash

import wandb

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import (
    _LRScheduler, StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
)

from torchmetrics.text.rouge import ROUGEScore
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup
)

from flash import Trainer
from flash.core.data.utils import download_data
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from flash.core.finetuning import NoFreeze
from flash.text import QuestionAnsweringData, QuestionAnsweringTask

from chaii import DATA_FOLDER_PATH

TRAIN_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "chaii_train.csv")
VAL_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "chaii_val.csv")

def split_dataset(filepath: str, fraction: float) -> None:
	df = pd.read_csv(filepath)

	# Splitting data into train and val beforehand since preprocessing will be different for datasets.
	tamil_examples = df[df["language"] == "tamil"]
	train_split_tamil = tamil_examples.sample(frac=fraction,random_state=200)
	val_split_tamil = tamil_examples.drop(train_split_tamil.index)

	hindi_examples = df[df["language"] == "hindi"]
	train_split_hindi = hindi_examples.sample(frac=fraction,random_state=200)
	val_split_hindi = hindi_examples.drop(train_split_hindi.index)

	train_split = pd.concat([train_split_tamil, train_split_hindi]).reset_index(drop=True)
	val_split = pd.concat([val_split_tamil, val_split_hindi]).reset_index(drop=True)

	train_split.to_csv(TRAIN_DATA_PATH, index=False)
	val_split.to_csv(VAL_DATA_PATH, index=False)