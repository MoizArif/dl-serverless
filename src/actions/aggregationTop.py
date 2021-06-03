import tensorflow as tf
import os, logging, time, resource
import numpy, json
from urllib.parse import urlparse
import redis

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

def main(args):

    print("INFO ::&> LOADING ALL THE MODELS FROM REDIS")
    params = args.get('meta')
    targets = args.get('target')
    cluster = args.get('cluster')
    level = int(args.get('level'))
    dataset = params['dataset']
    parsed = urlparse(params['db'])
    redis_client = redis.StrictRedis(
            host=parsed.hostname,
            port=parsed.port,
            decode_responses=True)

    all_models = []
    base_model = ''
    for model in targets:
        stored_json = json.loads(redis_client.execute_command('JSON.GET', model))
        curr_model = stored_json
        if "_0to" in model:
            base_model = tf.keras.models.model_from_json(stored_json)
            continue
        all_models.appends(curr_model)
        tf.keras.backend.clear_session()
        del curr_model

    print('Collected everything = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("INFO ::&> AVERAGING THE WEIGHTS OF ALL THE MODELS")
    avg_weights = list()
    for weights in zip(*all_models):
        avg_weights.append([numpy.array(values).mean(axis=0) for values in zip(*weights)])

    print('Built new weights = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("INFO ::&> COMPILING A NEW ENSEMBLE MODEL")
    base_model.set_weights(avg_weights)

    print("INFO ::&> STORING THE NEW MODEL")
    storage = 'ensemble_model_{0}'.format(params['file'])
    redis_client.execute_command('JSON.SET', storage, '.', json.dumps(base_model.to_json()))

    print('Total Memory = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    return {'Compute-Output': storage + ' stored successfully.'}
