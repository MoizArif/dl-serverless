import resource, codecs, redis
import os, time, sys, logging, json
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense,  Flatten
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.utils import get_file
from urllib.parse import urlparse
from math import ceil
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


def main(args):
    initTime = time.time()
    init_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("Loading and Preparing the Dataset.")
    params = args.get('meta')
    dataset = params['dataset']
    selected_model = params['model']
    batch_size = int(params['batch'])
    nb_epochs = int(params['epochs'])
    loss_delta = float(params['loss'])
    features = params['features']
    label = features['class']
    example = int(features['sample'])
    shape = features['shape']
    start, end = int(args.get('start')), int(args.get('end'))
    dtarange_start = example * (start) / 100
    dtarange_end = example * (end) / 100
    training_sample_size = int(dtarange_end - dtarange_start)

    print("MAKESPAN ::> Job submission -> Time = {0}.s | Memory = {1}.MB".format(time.time()-initTime, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("Loading and Preparing the Dataset.")
    train_ds = tfds.load(name=dataset, split=args.get('range'))
    train_ds = tfds.as_numpy(train_ds)
    train_X = np.asarray([im["image"] for im in train_ds])
    train_Y = np.asarray([im["label"] for im in train_ds])
    # train_X, train_Y = train_X[dtarange_start:dtarange_end], train_Y[dtarange_start:dtarange_end]

    classes = np.unique(train_Y)
    num_classes = len(classes)

    if shape[-1] < 3: #If is not RGB image
        train_X=np.dstack([train_X] * 3)
        # Reshape images as per the tensor format required by tensorflow
        train_X = train_X.reshape(-1, shape[0], shape[1], 3)
    print("no 1", train_X.shape)

    # Resize the images 48*48 required by models for datasets having smaller sizes
    if selected_model == 'inception':
        train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((75,75))) for im in train_X])
    else:
        if shape[0] < 48:
            train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_X])
        elif shape[0] > 56:
            train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((56,56))) for im in train_X])

    # Normalise the data and change data type
    train_X = train_X / 255.
    train_X = train_X.astype('float32')

    # Converting Labels to one hot encoded format
    train_label = to_categorical(train_Y)

    print("MAKESPAN ::> Processed data -> Time = {0}.s | Memory = {1}.MB".format(time.time()-initTime, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

    print("Preparing the Model")
    if selected_model == 'vgg16':
        from tensorflow.keras.applications import vgg16
        train_X = vgg16.preprocess_input(train_X)
        conv_base = vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(train_X.shape[1], train_X.shape[2], train_X.shape[3]))
    elif selected_model == 'resnet50':
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet import preprocess_input
        train_X = preprocess_input(train_X)
        conv_base = ResNet50(input_shape=(train_X.shape[1], train_X.shape[2], train_X.shape[3]), include_top=False, weights='imagenet', pooling='max')
    elif selected_model == 'resnet152':
        from tensorflow.keras.applications import ResNet152
        from tensorflow.keras.applications.resnet import preprocess_input
        train_X = preprocess_input(train_X)
        conv_base = ResNet152(input_shape=(train_X.shape[1], train_X.shape[2], train_X.shape[3]), include_top=False, weights='imagenet', pooling='max')
    elif selected_model == 'mobilenet':
        from tensorflow.keras.applications import MobileNet
        from tensorflow.keras.applications.mobilenet import preprocess_input
        train_X = preprocess_input(train_X)
        conv_base = MobileNet(input_shape=(train_X.shape[1], train_X.shape[2], train_X.shape[3]), include_top=False, weights='imagenet', pooling='max')
    elif selected_model == 'inception':
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        train_X = preprocess_input(train_X)
        conv_base = InceptionV3(input_shape=(train_X.shape[1], train_X.shape[2], train_X.shape[3]), include_top=False, weights='imagenet', pooling='max')
    else:
        print('The requested model is currently not supported.')
        sys.exit(0)

    print("MAKESPAN ::> Starting Training -> Time = {0}.s | Memory = {1}.MB".format(time.time()-initTime, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    train_features = conv_base.predict(np.array(train_X), batch_size=batch_size, verbose=2)
    for layer in conv_base.layers:
        layer.trainable = False

    # Saving the features so that they can be used for future
    np.savez("train_features", train_features, train_label)

    # Flatten extracted features
    train_features_flat = np.reshape(train_features, (training_sample_size, 1*1*train_features.shape[-1]))

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=(1*1*train_features.shape[-1])))
    model.add(layers.Dense(num_classes, activation='softmax'))

    print("Compiling the Model")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(),
        metrics=['acc'])

    # Incorporating early stopping for callback
    early_stopping = callbacks.EarlyStopping(
        monitor='loss',
        min_delta=loss_delta,
        patience=ceil(nb_epochs),
        mode='min',
        verbose=0,
        restore_best_weights=True)

    callback_list = [early_stopping]


    history = model.fit(
        train_features_flat,
        train_label,
        epochs=nb_epochs,
        callbacks=callback_list
    )

    print("MAKESPAN ::> Training completed -> Time = {0}.s | Memory = {1}.MB".format(time.time()-initTime, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("INFO ::&> SAVING THE TRAINED MODEL")
    model_info = model.to_json()
    print("INFO ::&> STORING THE TRAINED MODEL IN REDIS")
    storage = "model_{0}to{1}_{2}_{3}_{4}".format(start, end, dataset, selected_model, args.get('cid'))
    parsed = urlparse(params['db'])
    redis_client = redis.StrictRedis(
            host=parsed.hostname,
            port=parsed.port,
            decode_responses=True)
    redis_client.execute_command('JSON.SET', storage, '.', json.dumps(model_info))
    output = {'Compute-Output': storage + " trained and stored successfully."}
    print("Final Memory: {0}m".format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024) - init_memory))
    print("INFO ::&> TRAINING COMPLETED SUCCESSFULLY (*_*)")
    return output
