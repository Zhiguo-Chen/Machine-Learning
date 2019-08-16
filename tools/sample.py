import os 
import sys
import random
import math
import numpy as np 
import skimage.io
import matplotlib
import matplotlib.pyplot as plt 

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
from maskrcnn import model


def run():
    model.MaskRCNN()

if __name__ == "__main__":
    run()




