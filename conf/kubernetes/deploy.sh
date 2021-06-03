#!/usr/bin/env bash

# Deploying OpenWhisk
cd $HOME
kubectl create namespace openwhisk
helm init
git clone https://github.com/apache/openwhisk-deploy-kube.git
cd openwhisk-deploy-kube
helm install ./helm/openwhisk --namespace=openwhisk --name=owdev -f $HOME/dl-serverless/conf/kubernetes/mycluster.yaml

# WSK authentication configuration
printf "Enter Master\'s Public IP: "
read masterIP
wsk property set --apihost $masterIP:31001
wsk property set --auth 23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP


echo "Setting the path to binaries"
echo "alias disdel='python3 $HOME/dl-serverless/src/disdel.py'" >> $HOME/.bashrc
echo "alias whisk='python3 $HOME/dl-serverless/eval/defaultwhisk/whisk.py'" >> $HOME/.bashrc
