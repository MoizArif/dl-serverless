import tensorflow as tf
import os, logging, time, resource, sys
import codecs
import numpy
from urllib.parse import urlparse
import redis
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

def main(args):

    print("INFO ::&> LOADING ALL THE MODELS FROM REDIS")
    params = args.get('meta')
    dataset = params['dataset']
    parsed = urlparse(params['db'])
    redis_client = redis.StrictRedis(
            host=parsed.hostname,
            port=parsed.port,
            decode_responses=True)

    def getModels(storage, all_keys):
        all_models = list()
        for key in all_keys:
            print('Size of all_model : {0}'.format(sys.getsizeof(all_models)))
            model_data = redis_client.hget(storage, key)
            my_model = codecs.open("tf_model.h5", mode="w", encoding='ISO-8859-1')
            my_model.write(model_data)
            my_model.close()
            del my_model, model_data
            #print('Size ofmodel_data : {0}'.format(sys.getsizeof(model_data)))
            #print('Size of my_model : {0}'.format(sys.getsizeof(my_model)))
            #print('Size of tf_model = {0}'.format(os.stat('tf_model.h5').st_size))
            print('Memory at key {0} = {1}'.format(key, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            curr_model = tf.keras.models.load_model('tf_model.h5', compile=False)
            tf.keras.backend.clear_session()
            gc.collect()
            all_models.append(curr_model)
            del curr_model
        return all_models
    storage = '{0}_{1}'.format(params['model'], params['dataset'])
    all_keys = redis_client.hkeys(storage)
    while len(all_keys) < params['actions']:
        #print('Memory at length {0} = {1}'.format(len(all_keys), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        time.sleep(3)
        all_keys = redis_client.hkeys(storage)
    all_models = getModels(storage, all_keys)

    print('Collected everything = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("INFO ::&> AVERAGING THE WEIGHTS OF ALL THE MODELS")
    ''' The block of code below (Start to End) is built based on
        the model_weight_ensemble function from Jason Brownlee at
        https://machinelearningmastery.com/polyak-neural-network-model-weight-ensemble/
        ~~Start~~ '''
    number_of_layers = len(all_models[0].get_weights())
    number_of_model = params['actions']
    weights = [1/number_of_model for _ in range(1, number_of_model+1)]
    print('Building new weights = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    avg_model_weights = list()
    for layer in range(number_of_layers):
        # collect this layer from each model
        layer_weights = numpy.array([model.get_weights()[layer] for model in all_models])
        # weighted average of weights for this layer
        avg_layer_weights = numpy.average(layer_weights, axis=0, weights=weights)
        # store average layer weights
        avg_model_weights.append(avg_layer_weights)
        del layer_weights, avg_layer_weights
    '''
        ~~End~~ '''

    print('Building new model = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("INFO ::&> COMPILING A NEW ENSEMBLE MODEL")
    with tf.distribute.MirroredStrategy().scope():
        model = tf.keras.models.clone_model(all_models[0])
        model.set_weights(avg_model_weights)
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

    print("INFO ::&> STORING THE ENSEMBLE MODEL IN REDIS")
    model.save('final_model.h5')
    model_info = ""
    for line in codecs.open('final_model.h5', 'r', encoding='ISO-8859-1'):
        model_info += line
    model_name = dataset + '_' + params['model']
    print('Size of final_model = {0}'.format(os.stat('final_model.h5').st_size))
    redis_client.hset('ensemble', model_name, model_info)
    for key in redis_client.hkeys(storage):
        redis_client.hdel(storage, key)
    print('Total Memory = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    return {'SUMMARY':
                {
                    'Status':'Model trained successfully.',
                    'Last action': 'Ensemble model stored to Redis',
                    'Dataset': dataset
                }
            }
