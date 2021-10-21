from argparse import ArgumentParser
from dataclasses import asdict
from functools import partial

import os
import wandb

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from flash import Trainer
from flash.text import QuestionAnsweringData

from chaii.src.conf import WANDBLoggerConfiguration
from chaii.src.data import TRAIN_DATA_PATH, VAL_DATA_PATH, split_dataset
from chaii.src.model import ChaiiQuestionAnswering

EPOCHS = 10


def sweep_iteration(num_epochs: int, monitor: str, direction: str) -> float:
    with wandb.init() as run:
        config = run.config

        # We optimize the number of layers, hidden units in each layer and dropouts.
        batch_size = config.batch_size
        backbone = config.backbone
        learning_rate = config.learning_rate
        finetuning_strategy = config.finetuning_strategy
        optimizer = config.optimizer

        datamodule = QuestionAnsweringData.from_csv(
            train_file=TRAIN_DATA_PATH,
            val_file=VAL_DATA_PATH,
            batch_size=batch_size,
            backbone=backbone,
        )

        model = ChaiiQuestionAnswering(
            backbone=backbone,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=None,
            enable_ort=False,
        )

        # Setup Logger
        logger_configuration = WANDBLoggerConfiguration(
            group=f"{backbone}",
            job_type=f"sweep_{num_epochs}_epochs",
            name=f"{optimizer}_{learning_rate}_{finetuning_strategy}_{batch_size}",
            config=dict(
                batch_size=batch_size,
                backbone=backbone,
                learning_rate=learning_rate,
                finetuning_strategy=finetuning_strategy,
                optimizer=optimizer,
            ),
        )

        wandb_logger = WandbLogger(
            project="chaii-competition",
            log_model=True,
            **asdict(logger_configuration),
        )

        trainer = Trainer(
            logger=wandb_logger,
            checkpoint_callback=False,
            max_epochs=EPOCHS if num_epochs == 0 else num_epochs,
            gpus=1 if torch.cuda.is_available() else None,
        )

        try:
            trainer.finetune(
                model,
                datamodule=datamodule,
                strategy=finetuning_strategy,
            )
        except RuntimeError:
            if direction == "minimize":
                return wandb.log({monitor: 100.0})
            return wandb.log({monitor: -1.0})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--monitor", type=str, required=True)
    parser.add_argument(
        "-d", "--direction", type=str, required=True, choices=["maximize", "minimize"]
    )
    parser.add_argument("-t", "--trials", type=int, default=1, required=False)
    parser.add_argument("-e", "--epochs", type=int, default=0, required=False)
    parser.add_argument(
        "-s",
        "--sampler",
        type=str,
        default="random",
        required=False,
        choices=["random", "grid", "bayes"],
    )
    args = parser.parse_args()

    SWEEP_CONFIG = {
        "method": args.sampler,  # Random search
        "metric": {
            # We want to maximize val_accuracy
            "name": args.monitor,
            "goal": args.direction,
        },
        "parameters": {
            "backbone": {
                "values": [
                    "xlm-roberta-base",
                    "deepset/xlm-roberta-base-squad2",
                    "ai4bharat/indic-bert",
                ]
            },
            "batch_size": {"values": [4, 8, 16]},
            "learning_rate": {
                "distribution": "uniform",
                "min": 1e-8,
                "max": 1,
            },
            "finetuning_strategy": {"values": ["no_freeze", "freeze"]},
            "optimizer": {"values": ["adam", "adamw"]},
        },
    }

    seed_everything(42)
    if not (os.path.exists(TRAIN_DATA_PATH) and os.path.exists(VAL_DATA_PATH)):
        split_dataset()

    wandb.login()
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="chaii-competition")
    wandb.agent(
        sweep_id,
        function=partial(
            sweep_iteration,
            num_epochs=args.epochs,
            monitor=args.monitor,
            direction=args.direction,
        ),
        project="chaii-competition",
        count=args.trials,
    )
