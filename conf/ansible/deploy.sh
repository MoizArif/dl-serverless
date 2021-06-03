#!/usr/bin/env bash

echo "System Initialization in progress..."
sudo $HOME/dl-serverless/conf/ansible/init.sh

echo "Installing Apache OpenWhisk..."
bash $HOME/dl-serverless/conf/ansible/getWhisk.sh

echo "Setting the path to binaries"
echo -e "export PATH=$PATH:$HOME/openwhisk/bin" >> $HOME/.bashrc
echo "alias disdel='python3 $HOME/dl-serverless/src/disdel.py'" >> $HOME/.bashrc
echo "alias whisk='python3 $HOME/dl-serverless/eval/defaultwhisk/whisk.py'" >> $HOME/.bashrc

echo "(*_*) Congratulations! Your environment is ready."
