import logging
import sys
import os
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.utils import get_file
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
import tensorflow_datasets as tfds
import tensorflow.keras.datasets
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import importlib
import numpy as np  # linear algebra
import resource
import time
initTime = time.time()
init_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


def train(args):
    print("Processing parameters")
    dataset = args[0]
    selected_model = args[1]
    label = int(args[5])
    BATCH_SIZE = int(args[6])
    NB_EPOCHS = int(args[7])
    example = int(args[8])
    shape = []
    for index in range(3):
        shape.append(None if args[index + 2] ==
                     'None' else int(args[index + 2]))
    print(args)

    print("MAKESPAN ::> Job submission -> Time = {0}.s | Memory = {1}.MB".format(
        time.time() - initTime, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("Loading and Preparing the Dataset.")

    train_ds = tfds.load(name=dataset, split=tfds.Split.TRAIN)

    # tfds.as_numpy return a generator that yields NumPy array records out of a tf.data.Dataset
    train_ds = tfds.as_numpy(train_ds)
    train_X = np.asarray([im["image"] for im in train_ds])
    train_Y = np.asarray([im["label"] for im in train_ds])

    classes = np.unique(train_Y)
    num_classes = len(classes)

    if shape[-1] < 3:  # If is not RGB image

        train_X = np.dstack([train_X] * 3)

        # Reshape images as per the tensor format required by tensorflow
        train_X = train_X.reshape(-1, shape[0], shape[1], 3)
    print("no 1", train_X.shape)

    # Resize the images 48*48 required by models for datasets having smaller sizes
    if selected_model == 'inception':
        train_X = np.asarray([img_to_array(array_to_img(
            im, scale=False).resize((75, 75))) for im in train_X])
    else:
        if shape[0] < 48:
            train_X = np.asarray([img_to_array(array_to_img(
                im, scale=False).resize((48, 48))) for im in train_X])
        elif shape[0] > 56:
            train_X = np.asarray([img_to_array(array_to_img(
                im, scale=False).resize((56, 56))) for im in train_X])

    # Normalise the data and change data type
    train_X = train_X / 255.
    train_X = train_X.astype('float32')

    # Converting Labels to one hot encoded format
    train_label = to_categorical(train_Y)

    print("MAKESPAN ::> Processed data -> Time = {0}.s | Memory = {1}.MB".format(
        time.time() - initTime, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("Preparing the Model")

    #  Create base model
    if selected_model == 'vgg16':
        from tensorflow.keras.applications import vgg16
        train_X = vgg16.preprocess_input(train_X)
        conv_base = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(
            train_X.shape[1], train_X.shape[2], train_X.shape[3]))
    elif selected_model == 'resnet50':
        from tensorflow.keras.applications import ResNet50
        train_X = tensorflow.keras.applications.resnet.preprocess_input(
            train_X)
        conv_base = ResNet50(input_shape=(
            train_X.shape[1], train_X.shape[2], train_X.shape[3]), include_top=False, weights='imagenet', pooling='max')
    elif selected_model == 'resnet152':
        from tensorflow.keras.applications import ResNet152
        train_X = tensorflow.keras.applications.resnet.preprocess_input(
            train_X)
        conv_base = ResNet152(input_shape=(
            train_X.shape[1], train_X.shape[2], train_X.shape[3]), include_top=False, weights='imagenet', pooling='max')
    elif selected_model == 'mobilenet':
        from tensorflow.keras.applications import MobileNet
        train_X = tensorflow.keras.applications.mobilenet.preprocess_input(
            train_X)
        conv_base = MobileNet(input_shape=(
            train_X.shape[1], train_X.shape[2], train_X.shape[3]), include_top=False, weights='imagenet', pooling='max')
    elif selected_model == 'inception':
        from tensorflow.keras.applications import InceptionV3
        train_X = tensorflow.keras.applications.inception_v3.preprocess_input(
            train_X)
        conv_base = InceptionV3(input_shape=(
            train_X.shape[1], train_X.shape[2], train_X.shape[3]), include_top=False, weights='imagenet', pooling='max')
    else:
        print('The requested model is currently not supported.')
        sys.exit(0)
    # conv_base.summary()

    print("MAKESPAN ::> Starting Training -> Time = {0}.s | Memory = {1}.MB".format(
        time.time() - initTime, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    train_features = conv_base.predict(
        np.array(train_X), batch_size=BATCH_SIZE, verbose=2)
    for layer in conv_base.layers:
        layer.trainable = False

    # Saving the features so that they can be used for future
    np.savez("train_features", train_features, train_label)
    print(train_features.shape)

    # Flatten extracted features
    train_features_flat = np.reshape(
        train_features, (train_features.shape[0], 1 * 1 * train_features.shape[-1]))

    NB_TRAIN_SAMPLES = train_features_flat.shape[0]

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_dim=(1 * 1 * train_features.shape[-1])))
    model.add(layers.Dense(label, activation='softmax'))

    print("Compiling the Model")
    # Compile the model.
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(),
        metrics=['acc'])

    print("Training the Model")
    history = model.fit(
        train_features_flat,
        train_label,
        epochs=NB_EPOCHS
    )
    print("MAKESPAN ::> Training completed -> Time = {0}.s | Memory = {1}.MB".format(
        time.time() - initTime, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    final_memory = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024 - init_memory
    print("Memory: {0}m".format(final_memory))
    return final_memory
