from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import optuna

import torch
from torch.optim import Optimizer, Adam, AdamW
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from flash import Trainer
from flash.core.finetuning import _DEFAULTS_FINETUNE_STRATEGIES
from flash.text import QuestionAnsweringData

from chaii.src.data import TRAIN_DATA_PATH, VAL_DATA_PATH, split_dataset
from chaii.src.model import ChaiiQuestionAnswering

EPOCHS = 10


def objective(trial: optuna.trial.Trial, num_epochs: int) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    backbone = trial.suggest_categorical(
        "backbone",
        ["xlm-roberta-base", "deepset/xlm-roberta-base-squad2", "ai4bharat/indic-bert"],
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1)
    finetuning_strategy = trial.suggest_categorical(
        "finetuning_strategy", ["no_freeze", "freeze"]
    )

    datamodule = QuestionAnsweringData.from_csv(
        train_file=TRAIN_DATA_PATH,
        val_file=VAL_DATA_PATH,
        batch_size=batch_size,
        backbone=backbone,
    )

    model = ChaiiQuestionAnswering(
        backbone=backbone,
        learning_rate=learning_rate,
        optimizer=AdamW,
        enable_ort=True,
    )

    trainer = Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=EPOCHS if num_epochs == 0 else num_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="jaccard_score")],
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

    split_dataset()

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        partial(objective, num_epochs=args.epochs),
        n_trials=args.trials,
        timeout=600,
    )

    time = datetime.now()
    with open(f"optuna_hparams_search.txt_{time.strftime()}", "w") as f:
        f.write(f"Number of finished trials: {len(study.trials)}\n")
        f.write("Best trial:\n")
        trial = study.best_trial
        f.write(f">>> Value: {trial.value}\n")
        f.write(">>> Params:\n")
        for key, value in trial.params.items():
            f.write(f"   {key}: {value}\n")
