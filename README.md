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

Preprocessing takes some time to load (\~250 seconds when nothing has been tokenized so far and \~225 seconds when they already are and inside of their respective .npz files)
Fitting the model takes about \~(TBD), and loading an already trained model takes \~(TBD)

These numbers are from a computer with 16GB of ram and I believe not using a graphics card
There are also issues with training the model with ^^
