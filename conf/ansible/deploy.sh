#!/usr/bin/env bash

echo "System Initialization in progress..."
sudo bash ./init.sh

echo "Installing Apache OpenWhisk..."
bash ./getWhisk.sh

echo "Setting the path to binaries"
echo -e "export PATH=$PATH:~/openwhisk/bin" >> ~/.bashrc
echo "alias disdel='python3 ~/dl-serverless/src/disdel.py'" >> ~/.bashrc
echo "alias whisk='python3 ~/dl-serverless/eval/defaultwhisk/whisk.py'" >> ~/.bashrc

echo "(*_*) Congratulations! Your environment is ready."
