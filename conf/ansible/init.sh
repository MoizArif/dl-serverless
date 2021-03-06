#!/usr/bin/env bash

apt-get update
apt-get -y upgrade
apt-get -y autoremove
printf '[\u2713]\tSystem up-to-date\n'

# Installation of python and application dependencies
apt-get install -y git python3-pip python-setuptools build-essential libssl-dev libffi-dev python3-dev software-properties-common
pip3 install -r requirements.txt
printf '[\u2713]\tPython and application requirements installed\n'

# Installation of docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
apt-get update
apt-cache policy docker-ce
apt install -y docker-ce
systemctl is-active --quiet docker && echo Docker Service is running
#systemctl status docker

printf '[\u2713]\tDocker successfully installed\n'

# Installation of CouchDB
curl -L https://couchdb.apache.org/repo/bintray-pubkey.asc | apt-key add -
echo "deb https://apache.bintray.com/couchdb-deb bionic main" | tee -a /etc/apt/sources.list
apt-get update
apt-get install -y couchdb
#service couchdb status
#systemctl status couchdb
systemctl is-active --quiet couchdb && echo CouchDB Service is running
printf '[\u2713]\tCouchDB successfully installed\n'
