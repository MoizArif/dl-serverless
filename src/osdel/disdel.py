import sys
import argparse
import time, resource
from utils import Validator, Log
from controller import Controller

def main():
    init_time = time.time()
    init_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # Read commands parsed by users
    parser = argparse.ArgumentParser(prog='disdel', description='Distributed Serverless Deep Learning')
    parser.add_argument('file', type=str, help='Relative path to file containing dataset and model')
    parser.add_argument('-j', '--job', type=int, help='Batch or standalone computation')

    params = vars(parser.parse_args())
    disdel_control = Controller(params)
    disdel_control.configure().schedule().processRequest()

    final_time = time.time() - init_time
    final_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 - init_mem
    print('Job completed --> {0} @ Time: {1} & Memory: {2}'.format(params['file'], final_time, final_mem))

if __name__ == '__main__':
    main()
