#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Sentiment Analysis for Market Movement Prediction',
    version='1.0.0',
    description='Predicting the movements of the market with \
        sentiment analysis and historical prices',
    author='Bruce Nguyen',
    author_email='bruce.nguyen@aalto.fi',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/quan-possible/stock-nlp',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

