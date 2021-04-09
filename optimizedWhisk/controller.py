import sys
import time
import math
import random
import re
import json
import redis
import subprocess
import multiprocessing as mp
import concurrent.futures
from scheduler import Scheduler
from utils import Validator, Log, Ops
from urllib.parse import urlparse


class Controller():
    """Controller."""

    def __init__(self, params):
        self.image = "kevinassogba/tensorwhisk:latest"
        self.memory_limit = 70000  # 70 GB
        self.sys_timelimit = 900000  # 900 seconds
        self.nb_tr_action = 4 #1
        self.params = params
        self.u_cost = 0.000017

    def configure(self):
        # get features from manifest
        self.job = 0 if self.params['job'] is None else self.params['job']
        file_content = json.load(open(self.params['file']))
        # Build features
        self.model = file_content['model']
        self.params['features'] = file_content['dataset']
        self.params['dataset'] = self.params['features']['name']
        self.batch_size = file_content['args']['batch']
        self.epochs = file_content['args']['epoch']
        self.loss = file_content['args']['loss']
        self.cost = file_content['args']['cost']
        self.disdel_scheduler = Scheduler(self.params['features']['size'],
                                          self.params['features']['shape'],
                                          self.params['features']['class'],
                                          self.params['features']['sample'],
                                          self.model,
                                          self.batch_size,
                                          self.epochs)
        # Get db url for parameters storage
        dbIP = subprocess.run(
            ['/bin/bash', '-c', "docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' disdel-redis"], stdout=subprocess.PIPE).stdout.decode().strip()
        self.db_url = 'https://' + dbIP + ':6379'
        return self

    def schedule(self):
        # Training action memory computation
        self.training_memory = self.__estimateMemory(
            'training', self.nb_tr_action)
        print('TRAINING: Estimated #ofActions = {0} && Estimated Memory = {1}'.format(
            self.nb_tr_action, self.training_memory))

        # Time estimation
        max_time = math.ceil((self.cost / self.nb_tr_action) /
                             (self.u_cost * Ops().MBtoGB(self.training_memory)))
        self.time_limit = self.sys_timelimit if max_time > self.sys_timelimit else max_time
        if self.params['dataset'] == 'dmlab':
            self.time_limit = self.sys_timelimit
            if 'resnet' in self.model['name']:
                self.training_memory *= 1.8
            elif 'vgg16' in self.model['name']:
                self.training_memory = (self.training_memory / 6) * 2.9
            elif 'inception' in self.model['name']:
                self.training_memory = (self.training_memory / 3) * 1.8
        self.training_memory = int(self.training_memory)
        # Determine aggregation hierarchy
        self._scheduleAggregation()

        return self

    def _scheduleAggregation(self):
        # Determine maximum number of actions to fit in a single aggregation container
        # maximum number of actions to be processed by one aggregation container
        self.nb_ag_action = self.nb_tr_action
        self.__estimateMemory('aggregation', self.nb_ag_action)
        print('AGGREGATION: Estimated #ofActions = {0}'.format(
            self.nb_ag_action))

        self.aggregation_scheme = []
        inter_scheme = self._getScheme(self.nb_tr_action)
        self.aggregation_scheme.append(inter_scheme)
        while len(inter_scheme) > 1:
            inter_scheme = self._getScheme(low_level_action=len(inter_scheme))
            self.aggregation_scheme.append(inter_scheme)

    def _getScheme(self, low_level_action):
        # maximum number of aggregation containers to launch
        max_nb_ag = math.ceil(low_level_action / self.nb_ag_action)

        # Assign ids to low level actions
        # scheme -> [index_of_ag_container, nb_action_to_process, memory_of_ag_container]
        scheme = [[index, 0, 0] for index in range(max_nb_ag)]
        for _ in range(low_level_action):
            nb = random.randint(0, max_nb_ag - 1)
            while scheme[nb][1] > self.nb_ag_action - 1:
                nb += 1
            scheme[nb][1] += 1

        # Compute memory for clusters with assigned actions to process
        empty = []
        for ind in range(max_nb_ag):
            if scheme[ind][1] == 0:
                empty.append(ind)
            else:
                scheme[ind][2] = self.__estimateMemory(
                    'aggregation', scheme[ind][1])

        # Remove cluster with no assigned container
        for idx in empty:
            del scheme[idx]
        return scheme

    def __estimateMemory(self, task, nb_of_action):
        memory_estimate = self.__computeMemory(task, nb_of_action)

        if memory_estimate > self.memory_limit:
            return self.memory_limit
        while not Validator().validate(memory_estimate, self.memory_limit):  # while est >= limit
            Log.Warning().warnMemory(memory_estimate, self.memory_limit)
            gap = math.ceil((memory_estimate -
                             self.memory_limit) / self.memory_limit)
            if task == 'training':
                print('Deploying {0} containers'.format(self.nb_tr_action))
                self._incActionBy(task, gap)
                nb_of_action = self.nb_tr_action
            else:
                self._decActionBy(task, gap)
                nb_of_action = self.nb_ag_action
            memory_estimate = self.__computeMemory(task, nb_of_action)

        return memory_estimate

    def _incActionBy(self, task, inc_value):
        self.nb_tr_action += inc_value

    def _decActionBy(self, task, dec_value):
        self.nb_ag_action -= dec_value

    def __computeMemory(self, task, nb_of_action):
        job = 0 if task=='training' else 1
        print('job: ', job)
        return self.disdel_scheduler.estimateMemory(job, nb_of_action)

    def processRequest(self):
        # create package
        self.__createPackage()
        # Create and invoke training containers
        training_results = self.__train()
        # Create and invoke aggregation Services
        self.__aggregate(training_results)
        return self

    def __createPackage(self):
        metadata = json.dumps({"dataset": self.params['dataset'], "model": self.model['name'], "target": self.params['file'],
                               "batch": self.batch_size, "epochs": self.epochs, "loss": self.loss, "features": self.params["features"], "db": self.db_url})
        package_code = "wsk -i package update disdel-{0} --param meta '{1}'".format(
            self.job, metadata)
        self.__post(package_code)

    def __train(self):
        # Assume [[0,2,125], [1,3,235], [2,1,43]]
        size_gap = int(math.floor(100 / self.nb_tr_action))
        results = mp.Queue()
        pool = []
        # Proceed with training
        idx = 0
        for cluster in self.aggregation_scheme[0]:
            nb_of_action = cluster[1]
            for id in range(1, nb_of_action + 1):
                self.__createAction(
                    "training", self.training_memory, cluster[0], id)
                train_ops = mp.Process(target=self.invokeAction, args=(
                    "training", idx, size_gap, cluster[0], results))
                idx += 1
                pool.append(train_ops)
                train_ops.start()

        for action in pool:
            action.join()

        # process the returned results -
        # assign trained models to aggregation services, create and invoke them
        outcome = [results.get() for action in pool]
        return outcome

    def __createAction(self, name, memory, cluster, id=0):
        handle = name
        name += str(cluster) + '-' + str(id)
        code = "wsk -i action update disdel-{jid}/{task} ./{func}.py --docker {img} --memory {mem} --timeout {tim}".format(
            jid=self.job, task=name, func=handle, img=self.image, mem=memory, tim=self.time_limit)
        self.__post(code, debug_level=1)

    def invokeAction(self, name, action, target, cluster, results):
        if name == 'training':
            start = action * target
            end = (action + 1) * target
            name += str(cluster) + '-' + str(action + 1)
            data_range = 'train[{st}%:{nd}%]'.format(
                st=start, nd=end)
            print(
                "Training model on range '{0}-{1}'".format(start, end), end=" ")
            invocation = "wsk -i action invoke disdel-{jid}/{task} --param start {st} --param end {nd} --param range {rng} --param cid {clst}".format(
                jid=self.job, task=name, st=start, nd=end, rng=data_range, clst=cluster)
        else:
            print(target)
            name += str(action) + '-' + str(cluster)
            print(
                "Aggregation Services Level {0} Cluster {1} on-going....".format(action, cluster))
            invocation = "wsk -i action invoke disdel-{jid}/{task} --param target '{tgt}' --param cluster {clst} --param level {lvl}".format(
                jid=self.job, task=name, tgt=str(target), clst=cluster, lvl=action)
        response = self.__post(invocation, debug_level=1)
        final_output = self.__getResult(response)
        results.put(final_output)

    def __getResult(self, response):
        activation = "disdel"
        while "Compute-Output" not in response:
            if "error" in response and activation in response:
                response = self.__post(
                    "wsk -i activation result {0}".format(activation))
            elif "error" in response:
                print("\n", response, "\n")
                return "ExecutionError"
            else:
                activation = response.split(" ")[-1]
                response = self.__post(
                    "wsk -i activation result {0}".format(activation))
        print(response)
        return response

    def __post(self, command, debug_level=0):
        response = subprocess.run(['/bin/bash', '-c', command], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT).stdout.decode().strip()
        if debug_level is 1:
            print('${0}$....'.format(command))
            print('Result:', response)
        return response

    def __aggregate(self, lower_level_results):
        """
        Possible contents of the model info are: (1) "ExecutionError"
            (2) 'Compute-Output': storage + ' trained and stored successfully.'
        """
        # Determine number of aggregation level
        nb_ag_level = len(self.aggregation_scheme)
        stored_models = []
        # Assume one level is in the following form: [[0,2,125], [1,3,235], [2,1,43]]
        for level_idx in range(nb_ag_level):
            current_level = self.aggregation_scheme[level_idx]
            cluster_size = len(current_level)
            stored_models = self.__fetchModelName(
                cluster_size, lower_level_results)
            if cluster_size == 1 and level_idx == nb_ag_level:
                action_details = current_level[0]
                self.__createAction(
                    "aggregationTop", action_details[-1], level_idx, action_details[0])
                self.invokeAction("aggregationTop", level_idx, stored_models, action_details[0], [])
            else:
                lower_level_results=self.__aggregateMiddle(
                    level_idx, current_level, stored_models)

    def __fetchModelName(self, cluster_size, lower_level_results):
        # for each index, If the container has not failed, match the storage name
        # (reference of the model in the database)
        stored_models=[[] for _ in range(cluster_size)]
        # for model in iter(lower_level_results.get, None):
        for model in lower_level_results:
            if "Compute-Output" in model:
                storage=re.findall(r'\bmodel_\w+\d+', model)[0]
                cluster=int(storage.split('_')[-1])
                stored_models[cluster].append(storage)
        return stored_models

    def __aggregateMiddle(self, level_nb, level_details, stored_models):
        results=mp.Queue()
        pool=[]
        idx=0
        for cluster_details in level_details:
            target=stored_models[idx]
            idx += 1
            self.__createAction(
                "aggregationMiddle", cluster_details[-1], level_nb, cluster_details[0])
            ops=mp.Process(target = self.invokeAction, args = (
                "aggregationMiddle", level_nb, target, cluster_details[0], results))
            pool.append(ops)
            ops.start()

        for action in pool:
            action.join()

        outcome = [results.get() for action in pool]
        return outcome

    def invokeTrainingBkp(self, name, action, size_gap, cluster, results):
        start=action * size_gap
        end=(action + 1) * size_gap
        name += str(cluster) + '-' + str(action + 1)
        data_range='train[{st}%:{nd}%]'.format(
            st = start, nd = end)
        print(
            "Training model on range '{0}-{1}'".format(start, end), end = " ")
        invocation="wsk -i action invoke disdel-{jid}/{task} --param start {st} --param end {nd} --param range {rng} --param cid {clst}".format(
            jid = self.job, task = name, st = start, nd = end, rng = data_range, clst = cluster)
        response=self.__post(invocation, debug_level = 1)
        final_output=self.__getResult(response)
        results.put(final_output)
