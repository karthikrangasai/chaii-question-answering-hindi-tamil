import os
import pandas as pd

from chaii import DATA_FOLDER_PATH

TRAIN_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "chaii_train.csv")
VAL_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "chaii_val.csv")


def split_dataset(
    filepath: str = os.path.join(DATA_FOLDER_PATH, "train.csv"),
    fraction: float = 0.1,
) -> None:
    fraction = 1 - fraction
    df = pd.read_csv(filepath)

    # Splitting data into train and val beforehand since preprocessing will be different for datasets.
    tamil_examples = df[df["language"] == "tamil"]
    train_split_tamil = tamil_examples.sample(frac=fraction, random_state=200)
    val_split_tamil = tamil_examples.drop(train_split_tamil.index)

    hindi_examples = df[df["language"] == "hindi"]
    train_split_hindi = hindi_examples.sample(frac=fraction, random_state=200)
    val_split_hindi = hindi_examples.drop(train_split_hindi.index)

    train_split = pd.concat([train_split_tamil, train_split_hindi]).reset_index(
        drop=True
    )
    val_split = pd.concat([val_split_tamil, val_split_hindi]).reset_index(drop=True)

    train_split.to_csv(TRAIN_DATA_PATH, index=False)
    val_split.to_csv(VAL_DATA_PATH, index=False)
