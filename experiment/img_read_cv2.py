import cv2
import numpy as np
import matplotlib.pyplot as plt

imgPath = '../assets/test.jpeg'

img = cv2.imread(imgPath, 0)
print(img)
cv2.imshow('image', img)
cv2.waitKey()
# plt.imshow(img)
# plt.show()
