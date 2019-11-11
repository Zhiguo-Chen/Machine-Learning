import tensorflow as tf
from tensorflow import keras as kr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
import os

num_classes = 10
save_dir = os.path.join(os.path.abspath('.'), 'saved_model')
model_name = 'keras_cifar10_trained_model.h5'
fashion_minst = kr.datasets.fashion_mnist
cifar_data = kr.datasets.cifar10

# (train_images, train_labels), (test_images, test_labels) = fashion_minst.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print(train_images.shape)

(cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = cifar_data.load_data()

cifar_train_labels = kr.utils.to_categorical(cifar_train_labels, num_classes)
cifar_test_labels = kr.utils.to_categorical(cifar_test_labels, num_classes)

# print(cifar_train_images.shape)
# print(cifar_train_images.shape[0:])
print(cifar_train_images.shape[1:])
# print(cifar_train_images.shape[2:])
# print(cifar_train_images[0].shape)

# plt.figure()
# plt.imshow(cifar_train_images[10])
# plt.colorbar()
# plt.grid(False)
# plt.show()

input_shape = cifar_train_images.shape[1:]
inputs = Input(shape=input_shape)

layer1 = kr.layers.Conv2D(filters=32, kernel_size=3, strides=(
    1, 1), padding='same')(inputs)
activations1 = kr.layers.Activation('relu')(layer1)
layer2 = kr.layers.Conv2D(filters=32, kernel_size=3)(activations1)
activations2 = kr.layers.Activation('relu')(layer2)
max_pooling_layer1 = kr.layers.MaxPooling2D(pool_size=(2, 2))(activations2)
drop_out_layer1 = kr.layers.Dropout(0.25)(max_pooling_layer1)

layer3 = kr.layers.Conv2D(filters=64, kernel_size=3, padding='same')(drop_out_layer1)
activations3 = kr.layers.Activation('relu')(layer3)
layer4 = kr.layers.Conv2D(filters=64, kernel_size=3)(activations3)
activations4 = kr.layers.Activation('relu')(layer4)
max_pooling_layer2 = kr.layers.MaxPooling2D(pool_size=(2, 2))(activations4)
drop_out_layer2 = kr.layers.Dropout(0.25)(max_pooling_layer2)

flatten1 = kr.layers.Flatten()(drop_out_layer2)
dense1 = kr.layers.Dense(512)(flatten1)
dense_activation = kr.layers.Activation('relu')(dense1)
drop_out_layer3 = kr.layers.Dropout(0.5)(dense_activation)
dense2 = kr.layers.Dense(num_classes)(drop_out_layer3)
output = kr.layers.Activation('softmax')(dense2)

model = kr.models.Model(inputs=inputs, outputs=output)

model.compile(loss='categorical_crossentropy',
              optimizer=kr.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

cifar_train_images = cifar_train_images.astype('float32')
cifar_test_images = cifar_test_images.astype('float32')
cifar_train_images /= 255
cifar_test_images /= 255


model.fit(cifar_train_images, cifar_train_labels, batch_size=32, epochs=100,
          validation_data=(cifar_test_images, cifar_test_labels))

model_path = os.path.join(save_dir, model_name)
model.save(model_path)

scores = model.evaluate(cifar_test_images, cifar_test_labels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
