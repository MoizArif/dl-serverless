#!/usr/bin/env bash

# Install Apache OpenWhisk
cd $HOME
git clone https://github.com/apache/incubator-openwhisk.git openwhisk
cd openwhisk/
apt-get install -y nodejs-dev node-gyp libssl1.0-dev
apt-get install -y npm
(cd tools/ubuntu-setup && ./all.sh)
printf '[\u2713]\tSystem virtualization completed\n'

# Basic modificaition to the platform. Changing action memory, invoker memory, timeout and log limits
sed -ri "s/^(\s*)(invoker_heap | default('2g')).*/\1invoker_heap | default('8g')/" ./ansible/group_vars/all
sed -ri "s/^(\s*)(invoker_user_memory | default('2048m')).*/\1invoker_user_memory | default('71680m')/" ./ansible/group_vars/all
sed -ri 's/^(\s*)(max: "512m").*/\1max: "70000m"/' ./common/scala/src/main/resources/application.conf
sed -ri 's/^(\s*)(max: "5 m").*/\1max: "120 m"/' ./common/scala/src/main/resources/application.conf
sed -ri 's/^(\s*)(max: 1).*/\1max: 10/' ./common/scala/src/main/resources/application.conf


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
cd ..
apt-get install -y npm
./gradlew distDocker
cd ansible
ansible-playbook initdb.yml
ansible-playbook wipe.yml
ansible-playbook openwhisk.yml
ansible-playbook postdeploy.yml
docker run --name disdel-redis redislabs/rejson:latest
docker ps
printf '[\u2713]\tAnsible playbooks execution completed\n'
