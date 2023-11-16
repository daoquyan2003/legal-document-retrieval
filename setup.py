#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="An NLP project for Document Retrieval Task on ALQAC dataset.",
    author="",
    author_email="",
    url="https://github.com/daoquyan2003/legal-document-retrieval",
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "test_command = src.test:main",
        ]
    },
)
