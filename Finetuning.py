#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:13:18 2018

@author: zzq
"""

import numpy as np
import tensorflow as tf
import config as cfg
from VGG16 import Vgg16
def leak_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)

class Finetuning(object):
    def __init__(self,is_training = True):
        self.verbose = True
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]
        self.image_size = cfg.IMAGE_SIZE
        self.C = len(self.classes) # number of classes
        # offset for box center (top left point of each cell)
        self.batch_size = cfg.BATCH_SIZE
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.vgg = Vgg16()
        self.forward = self._build_net()
        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None,  self.C])
            self.loss_layer(self.forward, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
    def _build_net(self):
        """build the network"""
        if self.verbose:
            print("Start to build the network ...")
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        net = self.vgg.forward(self.images)
        net = self._flatten(net)
        net = self._fc_layer(net, 8, 4096, activation=leak_relu)
        net = self._fc_layer(net, 9, 4096, activation=leak_relu)
        net = self._fc_layer(net, 10, 4096, activation=leak_relu)
        net = self._fc_layer(net, 11, 512, activation=leak_relu) 
        net = self._fc_layer(net, 12, self.C)
        return tf.nn.sigmoid(net)

    def _conv_layer(self, x, id, num_filters, filter_size, stride):
        """Conv layer"""
    		#上一层的输出
        in_channels = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size,
                                                  in_channels, num_filters], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_filters,]))
        # padding, note: not using padding="SAME"
        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
		#1维 2 维 3维
        x_pad = tf.pad(x, pad_mat)
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
		#strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
        output = leak_relu(tf.nn.bias_add(conv, bias))
        if self.verbose:
            print("    Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s" \
                  % (id, num_filters, filter_size, stride, str(output.get_shape())))
        return output

    def _fc_layer(self, x, id, num_out, activation=None):
        """fully connected layer"""
        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.01))
        bias = tf.Variable(tf.zeros([num_out,]))  #逗号
        output = tf.nn.xw_plus_b(x, weight, bias)
        if activation:
            output = activation(output)
        if self.verbose:
            print("    Layer %d: type=Fc, num_out=%d, output_shape=%s" \
                  % (id, num_out, str(output.get_shape())))
        return output
    
    def loss_layer(self, predicts, labels):
        
    
        # class_loss
        self.class_delta = tf.abs(predicts - labels)
        self.predicts = predicts
        self.labels = labels
        print(predicts,labels,self.class_delta)
        class_loss = tf.reduce_sum(tf.square(self.class_delta),name='class_loss') 
        tf.losses.add_loss(class_loss)
        tf.summary.scalar('class_loss', class_loss)
    def _flatten(self, x):
        """flatten the x"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channle first mode
        nums = np.product(x.get_shape().as_list()[1:]) #乘积
        return tf.reshape(tran_x, [-1, nums])
    
if __name__ == "__main__":
    VGG_net = Vgg16()