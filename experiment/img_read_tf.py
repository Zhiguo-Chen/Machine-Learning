import tensorflow as tf
from img_path import img_path
import matplotlib.pyplot as plt

img = tf.io.read_file(img_path())
img = tf.image.decode_jpeg(img, channels=3)
plt.imshow(img)
plt.show()
