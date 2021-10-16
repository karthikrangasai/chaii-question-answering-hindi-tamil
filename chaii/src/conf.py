from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class ChaiiCompetitionConfiguration:
    # dataset specific
    train_val_split: float = field(default=0.1)
    batch_size: int = field(default=8)

    # model specific
    backbone: str = field(default="xlm-roberta-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # Optimizer and Scheduler specific
    learning_rate: float = field(default=2e-7)
    optimizer: str = field(default="adamw")
    scheduler: Optional[str] = field(default=None)

    # Training/Finetuning args
    operation_type: str = field(default="train")
    max_epochs: int = field(default=10)
    finetuning_strategy = "no_freeze"

    def __post_init__(self):
        assert self.operation_type in ["train", "finetune", "hparams_search"]


@dataclass
class WANDBLoggerConfiguration:
    group: str
    job_type: str
    name: str
    config: Dict[str, Any]
