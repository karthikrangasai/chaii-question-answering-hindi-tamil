from argparse import ArgumentParser
from functools import partial
import os
import time

import optuna
import pandas as pd

import torch
from torch.optim import Optimizer, Adam, AdamW
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning import seed_everything

from flash import Trainer
from flash.core.finetuning import _DEFAULTS_FINETUNE_STRATEGIES
from flash.text import QuestionAnsweringData
from chaii import OPTUNA_LOGS_PATH

from chaii.src.data import TRAIN_DATA_PATH, VAL_DATA_PATH, split_dataset
from chaii.src.model import ChaiiQuestionAnswering

EPOCHS = 10


def objective(
    trial: optuna.trial.Trial, num_epochs: int, monitor: str, direction: str
) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    backbone = trial.suggest_categorical(
        "backbone",
        ["xlm-roberta-base", "deepset/xlm-roberta-base-squad2", "ai4bharat/indic-bert"],
    )
    learning_rate = trial.suggest_uniform("learning_rate", 1e-8, 1)
    finetuning_strategy = trial.suggest_categorical(
        "finetuning_strategy", ["no_freeze", "freeze"]
    )
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])

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

    trainer = Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=EPOCHS if num_epochs == 0 else num_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=monitor)],
    )

    hyperparameters = dict(
        batch_size=batch_size,
        backbone=backbone,
        learning_rate=learning_rate,
        optimizer=optimizer,
        finetuning_strategy=finetuning_strategy,
    )
    trainer.logger.log_hyperparams(hyperparameters)

    try:
        trainer.finetune(
            model,
            datamodule=datamodule,
            strategy=finetuning_strategy,
        )
    except RuntimeError:
        if direction == "minimize":
            return 100.0
        return -1.0

    value = trainer.callback_metrics[monitor].item()
    return value


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
        choices=["random", "tpe"],
    )
    args = parser.parse_args()

    seed_everything(42)
    split_dataset()

    if args.sampler == "random":
        sampler: optuna.samplers.RandomSampler = optuna.samplers.RandomSampler(
            seed=int(time.time())
        )
    elif args.sampler == "tpe":
        sampler: optuna.samplers.TPESampler = optuna.samplers.TPESampler(
            seed=int(time.time())
        )

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction=args.direction, sampler=sampler, pruner=pruner
    )
    study.optimize(
        partial(
            objective,
            num_epochs=args.epochs,
            monitor=args.monitor,
            direction=args.direction,
        ),
        n_trials=args.trials,
        gc_after_trial=True,
    )

    trials_df: pd.DataFrame = study.trials_dataframe()
    trials_df.to_csv(
        path=os.path.join(
            OPTUNA_LOGS_PATH, f"{study._study_id}_{args.monitor}_{args.direction}.csv"
        )
    )

    with open(
        os.path.join(
            OPTUNA_LOGS_PATH,
            f"optuna_hparams_search_{study._study_id}_{args.monitor}_{args.direction}.txt",
        ),
        "w",
    ) as f:
        f.write(f"Number of finished trials: {len(study.trials)}\n")
        f.write("Best trial:\n")
        trial = study.best_trial
        f.write(f">>> Value: {trial.value}\n")
        f.write(">>> Params:\n")
        for key, value in trial.params.items():
            f.write(f"   {key}: {value}\n")
