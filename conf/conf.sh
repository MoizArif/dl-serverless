#!/usr/bin/env bash

printf 'Orchestration Engine (Ansible / Kubernetes)[Default: Ansible]: '
read engine
if [[ $engine != '' ]] && [[ $engine == 'Kubernetes' ]]
then
    echo "Kubernetes is currently not supported"
    sudo bash ./kubernetes/deploy.sh
elif [[ $engine != '' ]] && [[ $engine == 'Ansible' ]]
then
    echo "Orchestration Engine set to Ansible"
    sudo bash ./ansible/deploy.sh
else
    echo "Orchestration Engine set to Ansible"
    sudo bash ./ansible/deploy.sh
fi
