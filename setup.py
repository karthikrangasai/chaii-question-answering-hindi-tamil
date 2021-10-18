#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="chaii-question-answering",
    version="0.1.0",
    description="Question Answering on Hindi and Tamil Languages using Lightning Flash.",
    author="",
    author_email="",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/karthikrangasai/chaii-question-answering-hindi-tamil",
    install_requires=[
        "numpy",
        "pandas",
        "torch>=1.7.1",
        "pytorch-lightning>=1.4.0",
        "torchmetrics>=0.5.0",
        "wandb>=0.12",
        "optuna>=2.10.0",
        "torchtext>=0.10.0",
        "lightning-flash @ git+https://github.com/PyTorchLightning/lightning-flash.git#egg=lightning-flash[text]",
    ],
    packages=find_packages(),
)
