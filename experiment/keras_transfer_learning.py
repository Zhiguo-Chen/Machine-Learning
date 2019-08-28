import tensorflow as tf 
from tensorflow.python.keras import applications
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential, Model 
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from tensorflow import keras as K


img_width, img_height = 256, 256
train_data_dir = './data/train'
validation_dir = './data/val'
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 16
epochs = 50
weight_path = '/Users/hchen207/Documents/Excise/ML/mask-rcnn/mask_excise/Mask_excise/experiment/weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


model = applications.VGG19(weights=weight_path, include_top=False, input_shape=(img_width, img_height, 3))
print(model.output)
for layer in model.layers[:5]:
	layer.trainable = False
x = model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
pretiction = Dense(16, activation='softmax')(x)
model_final = Model(inputs=model.input, outputs=pretiction)
model_final.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])

