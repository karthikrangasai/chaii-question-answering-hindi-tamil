from argparse import ArgumentParser
from dataclasses import asdict

import wandb

from torch.optim import Optimizer, Adam, AdamW
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from flash import Trainer
from flash.core.utilities.types import OPTIMIZER_TYPE, LR_SCHEDULER_TYPE
from flash.text import QuestionAnsweringData


from chaii.src.conf import ChaiiCompetitionConfiguration, WANDBLoggerConfiguration
from chaii.src.data import TRAIN_DATA_PATH, VAL_DATA_PATH, split_dataset
from chaii.src.model import ChaiiQuestionAnswering


def train():
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--train_val_split", default=0.1, type=float)
    parser.add_argument("--batch_size", default=8, type=int)

    parser.add_argument(
        "--backbone", default="xlm-roberta-base", type=str, required=False
    )
    parser.add_argument("--learning_rate", default=2e-7, type=float)
    parser.add_argument("--optimizer", default="adamw", type=OPTIMIZER_TYPE)
    parser.add_argument("--scheduler", default=None, type=LR_SCHEDULER_TYPE)
    parser.add_argument("--operation_type", default="train", type=str)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument(
        "--finetuning_strategy", default="no_freeze", type=str, required=False
    )
    parser.add_argument("--debug", default=False, type=bool, required=False)

    args = parser.parse_args()

    config = ChaiiCompetitionConfiguration(
        train_val_split=args.train_val_split,
        batch_size=args.batch_size,
        backbone=args.backbone,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        operation_type=args.operation_type,
        max_epochs=args.max_epochs,
        finetuning_strategy=args.finetuning_strateg,
    )

    split_dataset()
    logger_configuration = WANDBLoggerConfiguration(
        group=f"{config.backbone}",
        job_type=f"{config.operation_type}_{config.max_epochs}_epochs",
        name=f"{config.optimizer}_{config.learning_rate}",
        config=asdict(config),
    )

    datamodule = QuestionAnsweringData.from_csv(
        train_file=TRAIN_DATA_PATH,
        val_file=VAL_DATA_PATH,
        batch_size=config.batch_size,
        backbone=config.backbone,
    )

    model = ChaiiQuestionAnswering(
        backbone=config.learning_rate,
        learning_rate=config.learning_rate,
        optimizer=AdamW,
        enable_ort=True,
    )

    # Setup Logger
    wandb_logger = WandbLogger(
        project="chaii-competition",
        log_model=True,
        **asdict(logger_configuration),
    )

    # Setup Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    earlystopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        filename="checkpoint/{epoch:02d}-{val_loss:.4f}",
        mode="max",
    )

    callbacks = [earlystopping, checkpoint_callback]

    if config.scheduler is not None:
        callbacks.append(lr_monitor)

    trainer = Trainer(
        fast_dev_run=args.debug,
        gpus=config.num_gpus,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        weights_summary="top",
        max_epochs=config.max_epochs,
    )

    wandb_logger.watch(model)

    if config.operation_type == "train":
        trainer.fit(model, datamodule=datamodule)
    elif config.operation_type == "finetune":
        trainer.finetune(
            model, datamodule=datamodule, strategy=config.finetuning_strategy
        )
    wandb.finish()


if __name__ == "__main__":
    train()
