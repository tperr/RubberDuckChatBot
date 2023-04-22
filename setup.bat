@echo off
echo setting up the environment...
conda env create -f environment.yml
echo setup complete, activating the environment
conda activate MARVIN
echo run "python rubberduck.py" to run the chatbot now