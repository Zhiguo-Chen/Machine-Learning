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
from utils.config import get_config
from utils.dic_to_obj import obj_dic


config_path = os.path.join(ROOT_DIR, 'config', 'config.json')
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


def update_config(config_obj):
    config_obj.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    config_obj.MINI_MASK_SHAPE = (56, 56)
    config_obj.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    config_obj.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    config_obj.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    return config_obj


def run():
    raw_config = get_config(config_path)
    config_class = obj_dic(raw_config)
    config_obj = config_class()
    config_obj = update_config(config_obj)
    print(type(config_class))
    print(type(config_obj))
    print(config_obj.BBOX_STD_DEV)
    model.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_obj)


if __name__ == "__main__":
    run()
