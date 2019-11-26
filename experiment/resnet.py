from tensorflow import keras as kr
import numpy as np
import os

save_dir = os.path.join(os.path.abspath('.'), 'saved_model')
model_name = 'keras_cifar10_resnet_model.h5'

num_classes = 10

cifar_data = kr.datasets.cifar10
(cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = cifar_data.load_data()
cifar_train_images = cifar_train_images.astype('float32')
cifar_test_images = cifar_test_images.astype('float32')
cifar_train_images /= 255
cifar_test_images /= 255
cifar_train_labels = kr.utils.to_categorical(cifar_train_labels, num_classes)
cifar_test_labels = kr.utils.to_categorical(cifar_test_labels, num_classes)


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    X = kr.layers.Conv2D(F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base +
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

    X = kr.layers.Add()([X, X_shortcut])
    X = kr.layers.Activation('relu')(X)

    return X


def convolutional_block(input_tensor, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    # X_shortcut = X
    print('origin X: ', input_tensor)

    X = kr.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s),
                         kernel_initializer=kr.initializers.glorot_uniform(seed=0), name=conv_name_base + '2a')(input_tensor)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = kr.layers.Activation('relu')(X)
    print('======')
    print(X)

    X = kr.layers.Conv2D(filters=F2, kernel_size=(f, f), padding='same',
                         kernel_initializer=kr.initializers.glorot_uniform(seed=0), name=conv_name_base + '2b')(X)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = kr.layers.Activation('relu')(X)
    print('++++++++')
    print(X)

    X = kr.layers.Conv2D(filters=F3, kernel_size=(1, 1),
                         kernel_initializer=kr.initializers.glorot_uniform(seed=0), name=conv_name_base + '2c')(X)
    X = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    print('---------')
    print(X)

    X_shortcut = kr.layers.Conv2D(filters=F3, kernel_size=(
        1, 1), strides=(s, s), padding='valid', kernel_initializer=kr.initializers.glorot_uniform(seed=0), name=conv_name_base + '1')(input_tensor)
    X_shortcut = kr.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    print(X)
    print(X_shortcut)
    X = kr.layers.Add()([X, X_shortcut])
    X = kr.layers.Activation('relu')(X)
    return X


def ResNet50(input_shape=(32, 32, 3), classes=10):
    X_input = kr.layers.Input(shape=input_shape)
    X = kr.layers.ZeroPadding2D((3, 3))(X_input)
    X = kr.layers.Conv2D(64, (7, 7), (2, 2), name='conv1',
                         kernel_initializer=kr.initializers.glorot_uniform(seed=0))(X)
    X = kr.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = kr.layers.Activation('relu')(X)
    X = kr.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, 3, [64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, 3, [128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, 3, [256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    print('avg pool <===>', X)
    X = convolutional_block(X, 3, [512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    print('avg pool ===>', X)
    # X = kr.layers.AveragePooling2D((2, 2),  name='avg_pool')(X)

    X = kr.layers.Flatten()(X)
    X = kr.layers.Dense(classes, activation='softmax', name='fc'+str(classes),
                        kernel_initializer=kr.initializers.glorot_uniform(seed=0))(X)
    model = kr.models.Model(inputs=X_input, outputs=X, name='ResNet50')
    return model


print(cifar_train_images.shape[1:])
print('===========>', cifar_train_images.shape)
model = ResNet50(cifar_train_images.shape[1:], classes=10)
model.compile(optimizer=kr.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6),
              loss='categorical_crossentropy', metrics=['accuracy'])


# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
# X_train = X_train_orig/255.
# X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T

print(cifar_train_labels.shape)


model.fit(cifar_train_images, cifar_train_labels, batch_size=32, epochs=100,
          validation_data=(cifar_test_images, cifar_test_labels))

model_path = os.path.join(save_dir, model_name)
model.save(model_path)

scores = model.evaluate(cifar_test_images, cifar_test_labels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
