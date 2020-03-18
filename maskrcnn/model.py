import tensorflow as tf
import datetime
import re
import os
from tensorflow import keras as kr


class MaskRCNN():
    def __init__(self, mode, config, model_dir):
        print('This is Mask RCNN model')
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.model = self.build(mode, config)
        print('============ init end ===============')

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
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception(
                'Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling. For example, use 256, 320, 384, 448, 512, ... etc')
        input_image = kr.layers.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name='input_image')
        input_image_meta = kr.layers.Input(
            shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        if mode == 'training':
            input_rpn_match = kr.layers.Input(
                shape=[None, 1], name='input_rpn_match', dtype=tf.int32)
            input_rpn_bbox = kr.layers.Input(
                shape=[None, 4], name='input_rpn_bbox', dtype=tf.float32)
            input_gt_class_ids = kr.layers.Input(
                shape=[None], name='input_gt_class_ids', dtype=tf.int32)
            input_gt_boxes = kr.layers.Input(
                shape=[None, 4], name='input_gt_boxes', dtype=tf.float32)
            gt_boxes = kr.layers.Lambda(lambda x: norm_boxes_graph(
                x, tf.shape(input_image)[1:3]))(input_gt_boxes)
            if config.USE_MINI_MASK:
                input_gt_masks = kr.layers.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None], name='input_gt_masks', dtype=bool)
            else:
                input_gt_masks = kr.layers.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], name='input_gt_masks', dtype=bool)
        elif mode == 'inference':
            input_anchors = kr.layers.Input(
                shape=[None, 4], name='input_anchors')

        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(
                input_image, stage5=True, train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(
                input_image=input_image, architecture=config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)

        print('============ build end ===============')
        return 1


def norm_boxes_graph(boxes, shape):
    print(boxes, ' === boxes ===')
    print(shape, ' ==== shape ====')
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    assert architecture in ['resnet50', 'resnet101']
    X = kr.layers.ZeroPadding2D((3, 3))(input_image)
    X = kr.layers.Conv2D(filters=64, kernel_size=(
        7, 7), strides=(2, 2), name='conv1', use_bias=True)(X)
    X = kr.layers.BatchNormalization(name='bn_conv1')(X)
    X = kr.layers.Activation('relu')(X)
    C1 = X = kr.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)
    X = conv_block(X, 3, [64, 64, 256], stage=2, block='a',
                   strides=(1, 1), train_bn=train_bn)
    X = identity_block(X, 3, [64, 64, 256], stage=2,
                       block='b', train_bn=train_bn)
    C2 = X = identity_block(X, 3, [64, 64, 256],
                            stage=2, block='c', train_bn=train_bn)
    X = conv_block(X, 3, [128, 128, 512], stage=3,
                   block='a', train_bn=train_bn)
    X = identity_block(X, 3, [128, 128, 512], stage=3,
                       block='b', train_bn=train_bn)
    X = identity_block(X, 3, [128, 128, 512], stage=3,
                       block='c', train_bn=train_bn)
    C3 = X = identity_block(
        X, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    X = conv_block(X, 3, [256, 256, 1024], stage=4,
                   block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        X = identity_block(X, 3, [256, 256, 1024],
                           stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = X
    if stage5:
        X = conv_block(X, 3, [512, 512, 2048], stage=5,
                       block='a', train_bn=train_bn)
        X = identity_block(X, 3, [512, 512, 2048],
                           stage=5, block='b', train_bn=train_bn)
        C5 = X = identity_block(
            X, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def conv_block(input_tensor, kernel_size, filter, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filter

    X = kr.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                         strides=strides, use_bias=use_bias, name=conv_name_base + '2a')(input_tensor)
    X = kr.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = kr.layers.Activation('relu')(X)

    X = kr.layers.Conv2D(filters=F2, kernel_size=(kernel_size, kernel_size),
                         padding='same', name=conv_name_base + '2b', use_bias=use_bias)(X)
    X = kr.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = kr.layers.Activation('relu')(X)

    X = kr.layers.Conv2D(filters=F3, kernel_size=(
        1, 1), name=conv_name_base + '2c', use_bias=use_bias)(X)
    X = kr.layers.BatchNormalization(name=bn_name_base + '2c')(X)

    X_shortcut = kr.layers.Conv2D(filters=F3, kernel_size=(
        1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    X_shortcut = kr.layers.BatchNormalization(
        name=bn_name_base + '1')(X_shortcut)

    X = kr.layers.Add()([X, X_shortcut])
    X = kr.layers.Activation('relu', name='res' +
                             str(stage) + block + '_out')(X)
    return X


def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    F1, F2, F3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X = kr.layers.Conv2D(filters=F1, kernel_size=(
        1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    X = kr.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = kr.layers.Activation('relu')(X)

    X = kr.layers.Conv2D(filters=F2, kernel_size=(
        kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=use_bias)(X)
    X = kr.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = kr.layers.Activation('relu')(X)

    X = kr.layers.Conv2D(filters=F3, kernel_size=(
        1, 1), name=conv_name_base + '2c', use_bias=use_bias)(X)
    X = kr.layers.BatchNormalization(name=bn_name_base + '3c')(X)

    X = kr.layers.Add()([X, input_tensor])
    X = kr.layers.Activation('relu', name='res' +
                             str(stage) + block + '_out')(X)
    return X
