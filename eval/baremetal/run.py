import json
import server
import time
import sys


file = open('$HOME/dl-serverless/eval/data_manifest.json')
full_data = json.load(file)
file.close()
model = sys.argv[1]
data = sys.argv[2]
batch = int(sys.argv[3])
epoch = int(sys.argv[4])

start_time = time.time()
features = full_data['datasets'][data]
shape = features['shape']
label = features['class']
example = features['example']
used_memory = server.train([data, model, shape[0], shape[1], shape[2], label, batch, epoch, int(example[0])])
run_time = time.time() - start_time
print("Dataset: {0} | Time: {1} sec | Memory: {2} MB".format(data, run_time, used_memory))
