#!/usr/bin/env bash

# This the main script running the connfiguration of the platform.
# It requires the user to specify which orchestration tool is desired.
# The default tool is Ansible.

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
