from tensorflow import keras as kr
import numpy as np


def identity_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    X = kr.layers.Conv2D(F1, (1, 1), kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base +
                         '2a', kernel_initializer=kr.initializers.glorot_uniform(seed=0))(X)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = kr.layers.Activation('relu')(X)

    X = kr.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                         name=conv_name_base + '2b', kernel_initializer=kr.initializers.glorot_uniform(seed=0))(X)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = kr.layers.Activation('relu')(X)

    X = kr.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=kr.initializers.glorot_uniform(seed=0))(X)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = kr.layers.Add()[X, X_shortcut]
    X = Kr.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X

    X = kr.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                         kernel_initializer=kr.initializers.glorot_uniform(seed=0), name=conv_name_base + '2a')(X)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = kr.layers.Activation('relu')(X)

    X = kr.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                         kernel_initializer=kr.initializers.glorot_uniform(seed=0), name=conv_name_base + '2b')(X)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = kr.layers.Activation('relu')(X)

    X = kr.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                         kernel_initializer=kr.initializers.glorot_uniform(seed=0), name=conv_name_base + '2c')(X)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = kr.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(
        s, s), padding='valid', kernel_initializer=kr.initializers.glorot_uniform(seed=0), name=conv_name_base + '1')(X_shortcut)
    X_shortcut = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    X = tf.layers.Add()([X, X_shortcut])
    return X
