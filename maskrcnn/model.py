import tensorflow as tf
import datetime
import re
import os
from tensorflow.keras import layers as KL


class MaskRCNN():
    def __init__(self, mode, config, model_dir):
        print('This is Mask RCNN model')
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.model = self.build(mode, config)

    def set_log_dir(self, model_path=None):
        self.epoch = 0
        now = datetime.datetime.now()
        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(
                    m.group(3)), int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)
        print(self.config.NAME, '=========')
        self.log_dir = os.path.join(
            self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))
        self.checkpoint_path = os.path.join(
            self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def build(self, mode, config):
        assert mode in ['training', 'inference']
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception('Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling. For example, use 256, 320, 384, 448, 512, ... etc')
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name='input_image')
        return 1
