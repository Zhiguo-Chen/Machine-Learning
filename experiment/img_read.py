import cv2
import numpy as np 

imgPath = '../assets/test.jpeg'

img = cv2.imread(imgPath, 0)
cv2.show('image', img)