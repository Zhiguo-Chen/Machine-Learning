from tensorflow import keras as kr
import os

pic_dir = os.path.join(os.path.abspath('.'), 'img')
pic_path = os.path.join(pic_dir, 'bag.jpeg')

img = kr.preprocessing.image.load_img(pic_path)
print(type(img))
# img.show()
img_array = kr.preprocessing.image.img_to_array(img)
print(img_array.shape)
img_pil = kr.preprocessing.image.array_to_img(img_array)
print(type(img_pil))
img_pil.show()
