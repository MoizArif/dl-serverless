import argparse
import subprocess
import sys
import json


def main():
    # Read commands parsed by users
    parser = argparse.ArgumentParser(
        prog='whisk', description='All in One action deep learning')
    parser.add_argument('model', type=str,
                        help='Target model for training job')
    parser.add_argument('dataset', type=str,
                        help='Target dataset (should be among tf.datasets)')
    parser.add_argument('-b', '--batch', type=int,
                        help='Size of data batch. \nDefault value is 64.')
    parser.add_argument('-e', '--epoch', type=int,
                        help='Number of epochs. \nDefault value is 100.')
    parser.add_argument('-m', '--memory', type=int,
                        help='Action memory.')
    parser.add_argument('-j', '--job', type=int,
                        help='Batch or standalone computation')

    params = vars(parser.parse_args())
    params['file'] = "../../examples/{0}/{1}.json".format(
        params['dataset'], params[model])
    build(params)


def build(params):
    # get data and model info from file
    with open(params['file']) as file:
        file_content = json.load(file)

    image = "kevinassogba/tensorwhisk:latest"
    # Get datasets parameters
    data_features = file_content['dataset']
    params['features'] = data_features
    size_attributes = data_features['size'].split(" ")
    size_value = int(float(size_attributes[0]))
    dataset = data_features['name']
    batch_size = 64 if params['batch'] is None else params['batch']
    epochs = 100 if params['epoch'] is None else params['epoch']
    memory = 65758 if params['memory'] is None else params['memory']
    timeout = 60000000
    metadata = json.dumps({"model": file_content['model']['name'], "dataset": dataset,
                           "batch": batch_size, "epochs": epochs, "features": params["features"]})
    #import pdb
    # pdb.set_trace()

    # Create package and action
    package_code = "wsk -i package update deep --param meta '{0}'".format(
        metadata)
    post(package_code)

    action_code = "wsk -i action update deep/learning learning.py --docker {2} --memory {0} --timeout {1}".format(
        memory, timeout, image)
    post(action_code)

    # Invoke action
    answer = post("wsk -i action invoke deep/learning --result")
    activation = "random"

    # Check if action has completed.
    # If successful, proceed
    # Otherwise, restart based on error message
    #import pdb
    # pdb.set_trace()
    while "SUMMARY" not in answer:
        if "error" in answer:
            if activation in answer:
                answer = post("wsk -i activation result " + activation)
            elif "exceeds allowed threshold" in answer:
                print("Status: Failed")
                sys.exit(1)
            else:
                if "time" in answer:
                    timeout += 10000  # increase by 10 seconds
                elif "memory" in answer:
                    # elif "The action did not produce a valid response and exited unexpectedly." in answer:
                    memory += 512  # increase by 256 MB
                action_code = "wsk -i action update deep/learning learning.py --docker {2} --memory {0} --timeout {1}".format(
                    memory, timeout, image)
                post(action_code)
                answer = post("wsk -i action invoke deep/learning --result")
        else:
            activation = answer.split(" ")[-1]
            act_code = "wsk -i activation result " + activation
            answer = post(act_code)

    if "SUMMARY" not in answer:
        print("Failed with unexpected termination status")
        sys.exit(1)


def post(command):
    print('Running ({0})....'.format(command))
    reply = subprocess.run(['/bin/bash', '-c', command], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT).stdout.decode().strip()
    if "error" in reply:
        # print('Error:', reply)
        pass
    else:
        print('Result:', reply)
    return reply


if __name__ == '__main__':
    main()
