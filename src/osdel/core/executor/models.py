class Model(object):

    def __init__(self, model_features, initial_shape, number_of_classes):
        self.initial_shape = initial_shape
        self.layers = model_features
        self.activations = 0
        self.trainable_parameters = 0
        self.misc_params = initial_shape[0] * initial_shape[1] * initial_shape[2]
        self.layers_with_trainable_paramters = ["conv", "depth"]
        self.dense_layers_nodes = [64, number_of_classes]

    def build(self, model_type='new', train_params=0, latest_shape=None):
        if 'type' in self.layers and self.layers['type'] == 'pre-trained':
            self.activations = int(self.layers['activations'])
            self.build_preTrained(self.layers['parameters'], self.layers['output'])
            return
        # Compute in accordance with the model configuration
        current_shape = self.initial_shape
        print('Layer\t\tInfo\t\tParams\t\tShape')
        for (lyerid, layer) in self.layers.items():
            lyer_mem = 1
            for (name, info) in layer.items():
                size_of_filters = info[0]
                number_of_filters = info[1]
                number_of_channels = current_shape[2]
                self.activations += current_shape[0]*current_shape[1]*current_shape[2]
                if name in self.layers_with_trainable_paramters:
                    self.trainable_parameters += getattr(Layer,
                                name + "Params")(number_of_channels, size_of_filters, number_of_filters)

                current_shape = getattr(Layer,
                                name + "Output")(current_shape, size_of_filters, number_of_filters)
                print('{0}\t\t{1} {2}\t\t{3}\t\t{4}'.format(lyerid, *info, self.trainable_parameters, current_shape))

        # Add two dense layers on top of the model
        for dense in self.dense_layers_nodes:
            lyerid = int(lyerid) + 1
            self.trainable_parameters += getattr(Layer, "denseParams")(dense, current_shape)
            current_shape = getattr(Layer, "denseOutput")(dense)
            print('{0}\t\t{1}\t\t{2}\t\t{3}'.format(lyerid, dense, self.trainable_parameters, current_shape))
            self.activations += current_shape

    def build_preTrained(self, train_params, current_shape):
        self.trainable_parameters += train_params
        # Add two dense layers on top of the model
        for dense in self.dense_layers_nodes:
            self.trainable_parameters += getattr(Layer, "denseParams")(dense, current_shape)
            current_shape = getattr(Layer, "denseOutput")(dense)
            print('Dense\t\t{0}\t\t{1}\t\t{2}'.format(dense, self.trainable_parameters, current_shape))
            self.activations += current_shape

class Layer():

    def convParams(number_of_channels, size_of_filters, number_of_filters):
        return (number_of_channels *
                      (size_of_filters ** 2) * number_of_filters) + number_of_filters


    def depthParams(number_of_channels, size_of_filters, number_of_filters):
        return number_of_channels * (size_of_filters ** 2) + number_of_channels * number_of_filters + number_of_filters

    def denseParams(number_of_output_nodes, number_of_input_nodes):
        return number_of_input_nodes * number_of_output_nodes + number_of_output_nodes

    def convOutput(dataset_shape, size_of_filters, number_of_filters):
        #return (dataset_shape[0] - size_of_filters + 1,
        #               dataset_shape[1] - size_of_filters + 1, number_of_filters)

        return (dataset_shape[0], dataset_shape[1], number_of_filters)

    def depthOutput(dataset_shape, size_of_filters, number_of_filters):
        return (dataset_shape[0], dataset_shape[1], number_of_filters)

    def poolOutput(dataset_shape, size_of_filters, number_of_filters):
        return (dataset_shape[0]/2, dataset_shape[1]/2, dataset_shape[2], 0)

    def avgOutput(dataset_shape, size_of_filters, number_of_filters):
        return (1, 1, dataset_shape[2], 0)

    def flattenOutput(dataset_shape, size_of_filters, number_of_filters):
        return dataset_shape[0] * dataset_shape[1] * dataset_shape[2]

    def denseOutput(output_shape):
        return output_shape
