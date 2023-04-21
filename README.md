# RubberDuckChatBot
Rubber duck chat bot for CS4811 Intro to AI

Training data comes from the following website:
https://huggingface.co/datasets/koutch/stackoverflow_python
make sure to download the zip archive and unzip it in the same path as rubberduck.py
There should be Answers.csv, Questions.csv, and Tags.csv unzipped if there isn't any of those it will fail
imports needed:
numpy
pandas
tensorflow

Preprocessing takes some time to load (\~520)
Fitting the model takes time to load too (\~36000 seconds (\~10 hours) untrained but its a near instant load when the rubbermodel.h5 file is there)

These numbers are from a computer with 16GB of ram and I believe not using a graphics card
There are also issues with training the model with ^^