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


class ChaiiQuestionAnswering(QuestionAnsweringTask):
    def __init__(
       self,
        backbone: str = "distilbert-base-uncased",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5,
        enable_ort: bool = False,
    ):
        super().__init__(
            backbone=backbone,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            enable_ort=enable_ort,
        )

    @staticmethod
    def jaccard(str1, str2): 
        a = set(str1.lower().split()) 
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
        
    def compute_metrics(self, generated_tokens, batch):
        scores = []
        for example in batch:
            predicted_answer = generated_tokens[example["example_id"]]
            target_answer = example["answer"]["text"][0] if len(example["answer"]["text"]) > 0 else ""
            scores.append(ChaiiQuestionAnswering.jaccard(predicted_answer, target_answer))

        # result = self.rouge.compute()
        result = {"jaccard_score": torch.mean(torch.tensor(scores))}
        return result
