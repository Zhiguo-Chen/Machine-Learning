import tensorflow as tf
import datetime
import re
import os


class MaskRCNN():
    def __init__(self, mode, config, model_dir):
        print('This is Mask RCNN model')
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()

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
        self.log_dir = os.path.join(
            self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))
        self.checkpoint_path = os.path.join(
            self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def build(self, mode, config):
        assert mode in ['training', 'inference']
