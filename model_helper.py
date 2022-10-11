import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python.keras.layers import Dropout

from resnet_for_cifar import *


def create_model(name, num_classes, input_shape):
    if name == "ResNet50v2":
        model = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
            classes=num_classes
        )
    elif name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None,
            classes=num_classes
        )
    elif name == "ResNet50v2-pre":
        # input_layer = Input(shape=(32, 32, 3))

        model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling="max",
            classes=num_classes
        )
        # add top layers
        out = model.output
        out = Flatten()(out)
        out = BatchNormalization()(out)
        out = Dense(256, activation='relu')(out)
        out = Dropout(0.3)(out)
        out = BatchNormalization()(out)
        out = Dense(128, activation='relu')(out)
        out = Dropout(0.3)(out)
        out = BatchNormalization()(out)
        out = Dense(64, activation='relu')(out)
        out = Dropout(0.3)(out)
        out = Dense(num_classes, activation='softmax')(out)
        # outputs = Flatten()(out)

        model = Model(inputs=model.inputs, outputs=out)

    else:
        raise Exception("Model is not defined.")

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.summary()

    return model
