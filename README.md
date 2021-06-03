# On Realizing Efficient Deep Learning Using Serverless Computing

Serverless computing is gaining rapid popularity as it reduces capital cost and enables quick application deployment without managing complex computing resources while improving application scalability.
This project enables the execution of data intensive computational task, e.g., deep learning (DL), on serverless infrastructures. Key features addressed are resource-level constraints, specifically fixed memory allocation and short task timeouts, imposed by serverless platforms and that lead to premature job failures.
We remove these constraints and develop an effective runtime framework that ensures that the appropriate amount of memory is allocated to containers for storing application data and a suitable timeout is selected for each job based on its complexity.

## Requirements

The current version of this project is tested in a cloud-based setup using Apache OpenWhisk and TensorFlow, respectively as serverless platform and deep learning framework.

It has been tested in the following environments:

* Ansible for deployments on a single machine
* Kubernetes for deployments in a cluster of 4 and 8 nodes

### Dependencies for Ansible deployment
The configuration for an Ansible deployment requires an implicit installation of the following software packages:

* Docker
* CouchDB
* Ansible
* OpenWhisk
* Redis

### Dependencies for Kubernetes deployment
The configuration of a Kubernetes cluster for this project requires an implicit installation of the following software packages:

* Docker
* Helm
* Kubernetes
* OpenWhisk
* Redis

**NOTE: You do NOT need to install packages individually. Run the following scripts from the main directory.**

## Quickstart

To orchestrate the project either with Ansible or Kubernetes, please ensure you have a python version >= 3.6.

```
$ python3 --version
```
For Ansible deployments, get the ip address of the machine's eno1np0 interface.

```
$ ifconfig
```
Install all requirements and build the serverless environment

```
$ make install
```

Follow the installation procedure. You will be required to provide answers to some interactive questions such as:

```
Orchestration Engine (Ansible / Kubernetes)[Default: Ansible]:
```
Press Enter to choose default Ansible option or Reply with either Ansible or Kubernetes to specify a choice.

Next, a window for CouchDB configuration will appear with requirements to provide the following information:
```
General type of CouchDB configuration: (Standalone / clustered/ none)
```
This framework is tested with ***Standalone***

```
CouchDB interface bind address:
```
This framework is tested with ***0.0.0.0***

```
Password for the CouchDB "admin" user:
```
Create a password for you CouchDB "admin" user.

To complete the configuration of CouchDB database, you will be prompted to:

```
Enter CouchDB's Public IP: <ip addr from ifconfig>
Enter CouchDB's Username: admin
Enter CouchDB's Password: <created admin user password>
```

When the setup completes, source the .bashrc file to start interacting with the framework.

```
$ source ~/.bashrc
```

To test our project:

```
$ disdel <task> <model> <dataset>
```
You can also run the following command to see the description and available options for the parameters.

```
$ disdel -h
```
