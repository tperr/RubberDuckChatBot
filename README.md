# RubberDuckChatBot
Rubber duck chat bot for CS4811 Intro to AI

Training data comes from the following website:
https://huggingface.co/datasets/koutch/stackoverflow_python
Once dataset is downloaded, unzip Answers.csv and Questions.csv into the folder.  The chatbot will not work without them.

Runs on python v3.6.13
Neccessary installations:
numpy
pandas
tensorflow
bs4

You can download the entire project from google drive (dataset to big for github):


Preprocessing takes some time to load (\~520)
Fitting the model takes time to load too (\~36000 seconds (\~10 hours) untrained but its a near instant load when the rubbermodel.h5 file is there)
## How to run:
Make sure you are in an environment where the required modules are installed and run "python rubberduck.py"
If you do not have an environment set up run setup.sh (linux systems) or setup.bat (windows) to set up and activate the environment


If you have any issues with anything contact anyone in our group via email:
Joshua Farr jsfarr@mtu.edu
Quentin Ross qcross@mtu.edu
Ransom Duncan rrduncan@mtu.edu
Tim Perr tlperr@mtu.edu