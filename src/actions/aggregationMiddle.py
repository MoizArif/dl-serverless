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
    targets = targets.replace('[','').replace(']','').replace('""', '').split(',')
    for model in targets:
        model = model.strip()
        print(model)
        stored_json = json.loads(redis_client.execute_command('JSON.GET', model))
        curr_model = tf.keras.models.model_from_json(stored_json).get_weights()
        all_models.append(curr_model)
        if "_0to" not in model:
            redis_client.execute_command('JSON.DEL', model)
        tf.keras.backend.clear_session()
        del curr_model

    print('Collected everything = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("INFO ::&> AVERAGING THE WEIGHTS OF ALL THE MODELS")
    avg_weights = list()
    for weights in zip(*all_models):
        avg_weights.append([numpy.array(values).mean(axis=0).tolist() for values in zip(*weights)])

    print('Built new weights = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("INFO ::&> STORING THE NEW MODEL")
    storage = 'model_average_{0}_{1}_{2}_{3}'.format(level, cluster, dataset, params['model'])
    redis_client.execute_command('JSON.SET', storage, '.', json.dumps(avg_weights))

    print('Total Memory = {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    return {'Compute-Output': storage + ' aggregated and stored successfully.'}
