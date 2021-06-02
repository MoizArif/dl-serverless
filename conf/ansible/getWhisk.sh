#!/usr/bin/env bash

# Install Apache OpenWhisk
git clone https://github.com/apache/incubator-openwhisk.git openwhisk
cd openwhisk/
(cd tools/ubuntu-setup && ./all.sh)
printf '[\u2713]\tSystem virtualization completed\n'

cd ansible
export OW_DB=CouchDB
export OW_DB_PROTOCOL=http
export OW_DB_PORT=5984

printf "Enter CouchDB\'s Public IP: "
read publicIP
printf "Enter CouchDB\'s Username: "
read admin
printf "Enter CouchDB\'s Password: "
read password
export OW_DB_HOST=$publicIP
export OW_DB_USERNAME=$admin
export OW_DB_PASSWORD=$password
printf '[\u2713]\tEnvironment variables defined\n'

ansible-playbook setup.yml
ansible-playbook prereq.yml
./gradlew distDocker
ansible-playbook initdb.yml
ansible-playbook wipe.yml
ansible-playbook openwhisk.yml
ansible-playbook postdeploy.yml


docker ps
ansible-playbook -i environments/local openwhisk.yml
cd ../bin
export PATH=$PATH:$PWD
printf '[\u2713]\tAnsible playbooks execution completed\n'
