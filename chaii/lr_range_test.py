from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime

import os

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from flash import Trainer
from flash.text import QuestionAnsweringData
from chaii import LR_RANGE_TEST_FIGS_PATH

from chaii.src.conf import WANDBLoggerConfiguration
from chaii.src.data import TRAIN_DATA_PATH, VAL_DATA_PATH, split_dataset
from chaii.src.model import ChaiiQuestionAnswering

num_epochs: int = 3
min_lr: float = 1e-8
max_lr: float = 1
batch_size: int = 8
backbone: str = "xlm-roberta-base"
optimizer: str = "adamw"


def range_test(
    num_epochs: int,
    min_lr: float,
    max_lr: float,
    batch_size: int,
    backbone: str,
    optimizer: str,
) -> None:
    datamodule = QuestionAnsweringData.from_csv(
        train_file=TRAIN_DATA_PATH,
        val_file=VAL_DATA_PATH,
        batch_size=batch_size,
        backbone=backbone,
    )

    model = ChaiiQuestionAnswering(
        backbone=backbone,
        optimizer=optimizer,
        lr_scheduler=None,
        enable_ort=False,
    )

    logger_configuration = WANDBLoggerConfiguration(
        group=f"{backbone}",
        job_type=f"lr_range_test",
        name=f"{optimizer}_{batch_size}",
        config=dict(
            batch_size=batch_size,
            backbone=backbone,
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
        max_epochs=num_epochs,
        gpus=1 if torch.cuda.is_available() else None,
    )

    def get_num_training_steps(datamodule, trainer) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset_size = len(datamodule.train_dataloader())
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (
            dataset_size // effective_batch_size
        ) * trainer.max_epochs
        return max_estimated_steps

    num_training = get_num_training_steps(datamodule, trainer)

    lr_finder = trainer.tuner.lr_find(
        model=model,
        datamodule=datamodule,
        early_stop_threshold=None,
        mode="linear",
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=num_training,
    )

    lr_finder = trainer.tuner.lr_find(
        model=model,
        datamodule=datamodule,
        early_stop_threshold=None,
        mode="linear",
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=100,
    )

    # Results can be found in
    new_lr = lr_finder.suggestion()
    print(f"Suggested Learning Rate: {new_lr}")

    # Plot with
    fig = lr_finder.plot(suggest=True)
    ax = fig.get_axes()
    ax.set_title(f"Range ({min_lr}, {max_lr}) - {num_training} : {new_lr}")
    fig.savefig(os.path.join(LR_RANGE_TEST_FIGS_PATH, f"range_search_{datetime.now()}"))

    return None


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--num_epochs", type=int, required=True, default=3)
    parser.add_argument("--min_lr", type=float, required=True, default=1e-8)
    parser.add_argument("--max_lr", type=float, required=True, default=1)
    parser.add_argument("--batch_size", type=int, required=True, default=8)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)

    args = parser.parse_args()

    seed_everything(42)
    if not (os.path.exists(TRAIN_DATA_PATH) and os.path.exists(VAL_DATA_PATH)):
        split_dataset()

    range_test(
        num_epochs=args.num_epochs,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        batch_size=args.batch_size,
        backbone=args.backbone,
        optimizer=args.optimizer,
    )
