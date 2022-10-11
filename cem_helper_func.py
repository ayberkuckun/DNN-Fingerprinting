import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from imagenet_helper import load_databatch
from model_helper import create_model
from resnet_for_cifar import *

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 100
data_augmentation = False


def prepare_data(dataset='cifar10', subtract_pixel_mean=True, shuffle=False, seed=None):
    if dataset == 'cifar10':
        num_classes = 10
        # Load the CIFAR-10 data.
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif dataset == 'imagenet32-pre':
        # Load the ImageNet32 data.
        num_classes = 1000
        img_size = 32
        img_size2 = img_size * img_size
        folder = "datasets/Imagenet32_train_npz/"

        # Subsample 100 classes?
        np.random.seed(105)
        permuted_labels = np.random.permutation(1000) + 1
        label1000 = permuted_labels[:1000]

        # Load train data.
        for idx in range(1, 11):
            if idx == 1:
                x_train, y_train = load_databatch(folder, idx, label1000)
            else:
                x_train_part, y_train_part = load_databatch(folder, idx, label1000)
                x_train = np.concatenate((x_train, x_train_part), axis=0)
                y_train = np.concatenate((y_train, y_train_part), axis=0)
        y_train = y_train.reshape((-1, 1))

        # Load validation data.
        val_ds = np.load("datasets/val_data.npz", allow_pickle=True)
        val_labels = val_ds['labels']
        val_data = val_ds['data']

        label_indexes = np.in1d(val_labels, label1000)
        val_labels = val_labels[label_indexes]
        val_data = val_data[label_indexes]

        # Labels are indexed from 1, shift it so that indexes start at 0
        val_labels = val_labels - 1

        x_test = np.dstack((val_data[:, :img_size2], val_data[:, img_size2:2 * img_size2], val_data[:, 2 * img_size2:]))
        x_test = x_test.reshape((x_test.shape[0], img_size, img_size, 3))
        y_test = val_labels.reshape((-1, 1))

    elif dataset == 'mnist':
        num_classes = 10
        # Load the MNIST data.
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = np.stack([x_train, x_train, x_train], axis=-1)
        x_test = np.stack([x_test, x_test, x_test], axis=-1)

        x_train = tf.image.resize(x_train, [32, 32]).numpy()
        x_test = tf.image.resize(x_test, [32, 32]).numpy()
    else:
        # Load the ImageNet32 data.
        num_classes = 100
        img_size = 32
        img_size2 = img_size * img_size
        folder = "datasets/Imagenet32_train_npz/"

        # Subsample 100 classes
        np.random.seed(105)
        permuted_labels = np.random.permutation(1000) + 1
        label100 = permuted_labels[:100]

        # Load train data.
        for idx in range(1, 11):
            if idx == 1:
                x_train, y_train = load_databatch(folder, idx, label100)
            else:
                x_train_part, y_train_part = load_databatch(folder, idx, label100)
                x_train = np.concatenate((x_train, x_train_part), axis=0)
                y_train = np.concatenate((y_train, y_train_part), axis=0)
        y_train = y_train.reshape((-1, 1))


        # Load validation data.
        val_ds = np.load("datasets/val_data.npz", allow_pickle=True)
        val_labels = val_ds['labels']
        val_data = val_ds['data']

        label_indexes = np.in1d(val_labels, label100)
        val_labels = val_labels[label_indexes]
        val_data = val_data[label_indexes]

        # Labels are indexed from 1, shift it so that indexes start at 0
        val_labels = val_labels - 1

        x_test = np.dstack((val_data[:, :img_size2], val_data[:, img_size2:2 * img_size2], val_data[:, 2 * img_size2:]))
        x_test = x_test.reshape((x_test.shape[0], img_size, img_size, 3))
        y_test = val_labels.reshape((-1, 1))

    # for i in range(10):
    #     plt.figure()
    #     plt.imshow(x_train[i])
    #     plt.show()

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    # Convert class vectors to binary class vectors.
    if dataset == 'imagenet32':
        for label_set in [y_train, y_test]:
            for no1, label1 in enumerate(label100 - 1):
                for no2, label2 in enumerate(label_set):
                    if label1 == label2:
                        label_set[no2] = no1

    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    if shuffle:
        if seed:
            np.random.seed(seed)

        idx_train = np.random.permutation(len(x_train))
        idx_test = np.random.permutation(len(x_test))

        x_train, y_train = x_train[idx_train], y_train[idx_train]
        x_test, y_test = x_test[idx_test], y_test[idx_test]

    # # Default
    # x_train = x_train.cache().map(
    # preprocess_dataset, num_parallel_calls=AUTOTUNE).cache().shuffle(
    # len(x_train), seed=seed).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    return (x_train, y_train), (x_test, y_test)


def preprocess_dataset():
    raise NotImplementedError


def create_resnet_model(input_shape, version=1, n=3, num_classes=10):
    # Computed depth from supplied model parameter n
    depth = 0
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Model name, depth and version
    model_name = 'ResNet%dv%d' % (depth, version)

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    # model.summary()

    return model, model_name


def fit_model(x_train, y_train, x_test, y_test, model, model_name, model_type, model_no, dataset_name):
    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), f'saved_models/{dataset_name}/{model_type}/{model_no}')
    model_file_name = '%s_model.h5' % model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_file_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   monitor='val_loss',
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15)

    # callbacks = [checkpoint, lr_reducer, early_stopping]
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)

    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    return model


def prepare_models(x_train, y_train, x_test, y_test, dataset, model, input_shape, use_saved, surrogate_no, reference_no):
    surrogate_models = []
    reference_models = []

    if dataset == 'cifar10' or dataset == 'mnist':
        num_classes = 10
    elif dataset == 'imagenet32-pre':
        num_classes = 1000
    else:
        num_classes = 100

    if use_saved:
        # # Pre-trained models.
        # if model == "ResNet50v2-pre":
        #     # Create source model
        #     source_model = create_model(model, num_classes, input_shape)
        #
        #     # Create surrogate models
        #     for i in range(surrogate_no):
        #         surrogate_model = create_model(model, num_classes, input_shape)
        #         surrogate_models.append(surrogate_model)
        #
        #     # Create reference models
        #     for i in range(reference_no):
        #         reference_model = create_model(model, num_classes, input_shape)
        #         reference_models.append(reference_model)
        #
        #     return source_model, reference_models, surrogate_models

        path_source = os.path.join(os.getcwd(), f'saved_models/{dataset}/source/0/{model}_model.h5')
        source_model = keras.models.load_model(path_source)

        for i in range(surrogate_no):
            path_surrogate = os.path.join(os.getcwd(), f'saved_models/{dataset}/surrogate/{i+1}/{model}_model.h5')
            surrogate_model = keras.models.load_model(path_surrogate)
            surrogate_models.append(surrogate_model)

        for i in range(reference_no):
            path_reference = os.path.join(os.getcwd(), f'saved_models/{dataset}/reference/{i+1}/{model}_model.h5')
            reference_model = keras.models.load_model(path_reference)
            reference_models.append(reference_model)

    else:
        if model == "ResNet20v1":
            # # Create source model
            # source_model, source_model_name = create_resnet_model(input_shape=input_shape, version=1, n=3, num_classes=num_classes)
            #
            # # Fit source model
            # print(f"Fitting source model")
            # fit_model(x_train, y_train, x_test, y_test,
            #           model=source_model,
            #           model_name=source_model_name,
            #           model_type='source',
            #           model_no=0,
            #           dataset_name=dataset)

            path_source = os.path.join(os.getcwd(), f'saved_models/{dataset}/source/0/{model}_model.h5')
            source_model = keras.models.load_model(path_source)

            # Get labels for surrogate models
            y_predict_train_source = source_model.predict(x_train)
            y_predict_test_source = source_model.predict(x_test)

            # Create surrogate models
            surrogate_model_name = ''
            for i in range(surrogate_no):
                surrogate_model, surrogate_model_name = create_resnet_model(input_shape=input_shape, version=1, n=3, num_classes=num_classes)
                surrogate_models.append(surrogate_model)

            # Fit surrogate models
            for no, s_model in enumerate(surrogate_models):
                print(f"Fitting surrogate No: {no}")

                fit_model(x_train, y_predict_train_source, x_test, y_predict_test_source,
                          model=s_model,
                          model_name=surrogate_model_name,
                          model_type='surrogate',
                          model_no=no,
                          dataset_name=dataset)

            # Create reference models
            reference_model_name = ''
            for i in range(reference_no):
                reference_model, reference_model_name = create_resnet_model(input_shape=input_shape, version=1, n=3, num_classes=num_classes)
                reference_models.append(reference_model)

            # Fit reference models
            for no, r_model in enumerate(reference_models):
                print(f"Fitting reference No: {no}")

                fit_model(x_train, y_train, x_test, y_test,
                          model=r_model,
                          model_name=reference_model_name,
                          model_type='reference',
                          model_no=no,
                          dataset_name=dataset)

        else:
            source_model_name = reference_model_name = surrogate_model_name = model

            # Create source model
            source_model = create_model(model, num_classes, input_shape)

            # Fit source model
            print(f"Fitting source model")
            fit_model(x_train, y_train, x_test, y_test,
                      model=source_model,
                      model_name=source_model_name,
                      model_type='source',
                      model_no=0,
                      dataset_name=dataset)

            # path_source = os.path.join(os.getcwd(), f'saved_models/{dataset}/source/0/{model}_model.h5')
            # source_model = keras.models.load_model(path_source)

            # Get labels for surrogate models
            y_predict_train_source = source_model.predict(x_train)
            y_predict_test_source = source_model.predict(x_test)

            # Create surrogate models
            for i in range(surrogate_no):
                surrogate_model = create_model(model, num_classes, input_shape)
                surrogate_models.append(surrogate_model)

            # Fit surrogate models
            for no, s_model in enumerate(surrogate_models):
                print(f"Fitting surrogate No: {no}")

                fit_model(x_train, y_predict_train_source, x_test, y_predict_test_source,
                          model=s_model,
                          model_name=surrogate_model_name,
                          model_type='surrogate',
                          model_no=no,
                          dataset_name=dataset)

            # Create reference models
            for i in range(reference_no):
                reference_model = create_model(model, num_classes, input_shape)
                reference_models.append(reference_model)

            # Fit reference models
            for no, r_model in enumerate(reference_models):
                print(f"Fitting reference No: {no}")

                fit_model(x_train, y_train, x_test, y_test,
                          model=r_model,
                          model_name=reference_model_name,
                          model_type='reference',
                          model_no=no,
                          dataset_name=dataset)

    return source_model, reference_models, surrogate_models
