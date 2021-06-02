#!/usr/bin/env bash

echo "System Initialization in progress..."
sudo bash ./init.sh

echo "Installing Apache OpenWhisk..."
bash ./getWhisk.sh

echo "(*_*) Congratulations! Your environment is ready."
