from tensorflow import keras as kr
import numpy as np
import os

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
save_dir = os.path.join(os.path.abspath('.'), 'saved_model')
model_name = 'keras_cifar10_resnet_model.h5'
img_dir = os.path.join(os.path.abspath('.'), 'img')
img_path = os.path.join(img_dir, 'hybrid-car-ch.jpg')
test_img = kr.preprocessing.image.load_img(img_path, target_size=(32, 32))
img_arry = kr.preprocessing.image.img_to_array(test_img)
img_arry = np.expand_dims(img_arry, axis=0)

model_path = os.path.join(save_dir, model_name)
model = kr.models.load_model(model_path)
prediction = model.predict(img_arry)
print(prediction[0])
print(class_names[np.argmax(prediction[0])])
