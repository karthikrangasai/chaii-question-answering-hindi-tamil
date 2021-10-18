from argparse import ArgumentParser
from functools import partial

import wandb

import torch
from pytorch_lightning import seed_everything

from flash import Trainer
from flash.text import QuestionAnsweringData

from chaii.src.data import TRAIN_DATA_PATH, VAL_DATA_PATH, split_dataset
from chaii.src.model import ChaiiQuestionAnswering

EPOCHS = 10


def sweep_iteration(num_epochs: int) -> float:
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
            enable_ort=False,
        )

        trainer = Trainer(
            logger=True,
            checkpoint_callback=False,
            max_epochs=EPOCHS if num_epochs == 0 else num_epochs,
            gpus=1 if torch.cuda.is_available() else None,
        )

        hyperparameters = dict(
            batch_size=batch_size,
            backbone=backbone,
            learning_rate=learning_rate,
            finetuning_strategy=finetuning_strategy,
        )
        trainer.logger.log_hyperparams(hyperparameters)

        trainer.finetune(
            model,
            datamodule=datamodule,
            strategy=finetuning_strategy,
        )

        jaccard_score = trainer.callback_metrics["jaccard_score"].item()
        return jaccard_score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--trials", type=int, default=1, required=False)
    parser.add_argument("-e", "--epochs", type=int, default=0, required=False)
    args = parser.parse_args()

    SWEEP_CONFIG = {
        "method": "random",  # Random search
        "metric": {
            # We want to maximize val_accuracy
            "name": "jaccard_score",
            "goal": "maximize",
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
            "lr": {
                "distribution": "uniform",
                "min": 1e-8,
                "max": 1,
            },
            "finetuning": {"values": ["no_freeze", "freeze"]},
            "optimizer": {"values": ["adam", "adamw"]},
        },
    }

    seed_everything(42)
    split_dataset()

    wandb.login()
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="chaii-competition")
    wandb.agent(
        sweep_id,
        function=partial(sweep_iteration, num_epochs=args.epochs),
        project="chaii-competition",
        count=args.trials,
    )
