import math
from models import Model


class Scheduler():
    """
        The Scheduler aggregates the number of parameters at each layer,
        the size of docker image and accessories space to estimate
        the memory utilization of the training job.
    """

    def __init__(self,
                 dataset_size, #input_size,
                 dataset_shape, #input_shape,
                 number_of_output_classes, #num_class,
                 number_of_data_records, #num_sample,
                 model_features,
                 batch=64,
                 epoch=5):

        self.dataset_size = self.__convertToMB(dataset_size)
        self.number_of_output_classes = number_of_output_classes
        self.number_of_data_records = number_of_data_records
        self.dataset_shape = dataset_shape
        self.model = model_features
        self.batch = batch
        self.epoch = epoch
        self.params = self.__parameterCount() #train_params, activation, misc
        print('There are {0} parameters'.format(self.params[0]))

    def estimateMemory(self, job=0, number_of_action=1):

        package_size = 225  # 225 MB for the docker image tensorwhisk
        # The size is factored by 3 because it is: (1) Downloaded (2) Loaded (3) Cached
        data = 3*self.dataset_size
        sft_limit = 512
        #data = self.dataset_size
        model_size = (self.params[0] * 4)/2**20
        # num_steps = self.number_of_data_records[job] / (self.batch * number_of_action)

        if job is 0:
            # training
            fit_estimate = self.getTrainingMemory(number_of_action)
            if self.model['name']=='vgg16':
                return math.ceil(data + fit_estimate + package_size + model_size + sft_limit) * 6
            elif self.model['name']=='inception':
                return math.ceil(data + fit_estimate + package_size + model_size + sft_limit) * 3
        elif job is 1:
            # Weights collection
            ag_estimate = self.getAggregationMemory(number_of_action)
            return ag_estimate + package_size + sft_limit
        else:
            # Inference
            fit_estimate = self.getInferenceMemory(number_of_action)

        #print('data: {0} | fit: {1} | package: {2}'.format(data, fit_estimate, package_size))
        print('data: {0} | fit: {1} | package: {2} | model: {3}'.format(data, fit_estimate, package_size, model_size))
        return math.ceil(data + fit_estimate + package_size + model_size + sft_limit)

    def getTrainingMemory(self, number_of_action):
        # Training
        activation_memory = self.getActivationMemory()
        params_memory = self.getParameterMemory('training')
        misc_memory = self.getMiscMemory()
        total_memory = (activation_memory * 2 * (self.batch/number_of_action)) + params_memory + misc_memory*self.epoch
        print("Trainable Mem: {0}\nActivations Mem: {1}\nMiscelaneous Mem: {2}".format(
                params_memory, activation_memory, misc_memory)
            )
        return total_memory

    def getParameterMemory(self, type):
        number_of_pass = 2
        if type is 'inference':
            return (self.params[0] * 4) / 2**20
        return (self.params[0] * (number_of_pass + 2) * 4) / 2**20

    def getActivationMemory(self):
        number_of_pass = 2  # forward and backward
        return (self.params[1] * number_of_pass * 4) / 2**20

    def getMiscMemory(self):
        return (self.params[2] * self.batch * 4) / 2**20

    def __convertToMB(self, size):
        size_attributes = size.split(" ")
        size_value = int(float(size_attributes[0]))
        if size_attributes[1] == "KiB":
            return size_value / 1024
        elif size_attributes[1] == "GiB":
            return size_value * 1024
        return size_value

    def __parameterCount(self):
        model = Model(self.model, self.dataset_shape, self.number_of_output_classes)
        model.build()
        print("Trainable parameters: {0}\nActivations: {1}\nMiscelaneous: {2}".format(
                model.trainable_parameters, model.activations, model.misc_params)
            )
        return (model.trainable_parameters,
                model.activations,
                model.misc_params)

    def getAggregationMemory(self, nb_of_target):
        model_memory = (self.params[0] * 4)/2**20
        ag_estimate = math.ceil(model_memory * (nb_of_target + 4))
        return ag_estimate


'''
    Illustration of number of parameters computation (MNIST)

    Layers                  Input                   Output                  #ofParams
    __init                  28x28 GreyScale         (28,28,1)               None
    Conv2D                  (28,28,1)               (26,26,32)              320
    MaxPool2D               (26,26,32)              (13,13,32)              None
    Flatten                 (13,13,32)              (5408)                  None
    Dense(64)               (5408)                  (64)                    346112
    Dense(10)               (64)                    (10)                    640
                                                                        -----------------
    Flatten                 (13,13,32)              (5408)                  None
    Dense(64)               (5408)                  (64)                    346112
'''
