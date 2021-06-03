# Performance Evaluation

To evaluate the performance of the framework, we evaluate it against a Baremetal Tensorflow deployment and Default serverless TensorFlow deployment to show that it eliminates DL job failures and reduces memory consumption and training time.

For Baremetal Tensorflow, first intall tensorflow and dependencies locally.
From the **eval/** folder:

```
$ pip3 install -r requirements.txt
$ cd baremetal
$ python3 run.py <model> <dataset> <batch> <epoch>
```

For Default serverless TensorFlow

```
$ cd defaultwhisk
$ whisk <model> <dataset> <batch> <epoch>
```
You can also run the following command to see the description and available options for parameters.

```
$ whisk -h
```
