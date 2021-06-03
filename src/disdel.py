import sys
import argparse
import time
import resource
from utils import Validator, Log
from controller import Controller


def main():
    init_time = time.time()
    init_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # Read commands parsed by users
    parser = argparse.ArgumentParser(
        prog='disdel', description='Optimized Serverless Deep Learning')
    parser.add_argument('model', type=str,
                        help='Target model for training job')
    parser.add_argument('dataset', type=str,
                        help='Target dataset (should be among tf.datasets)')
    parser.add_argument('-b', '--batch', type=int,
                        help='Size of data batch. \nDefault value is 64.')
    parser.add_argument('-e', '--epoch', type=int,
                        help='Number of epochs. \nDefault value is 5.')
    parser.add_argument('-j', '--job', type=int,
                        help='Job ID. Used for batch jobs to denote jobs.')

    params = vars(parser.parse_args())
    params['file'] = "../examples/{0}/{1}.json".format(
        params['dataset'], params[model])
    disdel_control = Controller(params)
    disdel_control.configure().schedule().processRequest()

    final_time = time.time() - init_time
    final_mem = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024 - init_mem
    print('Job completed --> {0} @ Time: {1} & Memory: {2}'.format(
        params['file'], final_time, final_mem))


if __name__ == '__main__':
    main()
