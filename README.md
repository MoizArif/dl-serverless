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

To setup up the environment as we did,

**You do NOT need to install packages individually. Run the following scripts from the main directory.**

#### Note
While installing CouchDB, a windows requiring an ip address for database configuration. The default value appearing on screen is 127.0.0.1. Change that value to 0.0.0.0

## Quickstart

To orchestrate the project either with Ansible or Kubernetes, please ensure you have a python version >= 3.6.

```
$ python3 --version
```

Install all requirements and build the serverless environment

```
$ make install
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
