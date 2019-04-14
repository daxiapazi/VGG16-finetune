#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 10:31:36 2018

@author: zzq
"""

import numpy as np
import tensorflow as tf
import cv2
import os
from Finetuning import Finetuning
import config as cfg
from visualization import plt_bboxes

class test(object):
    def __init__(self):
        self.verbose = True
        # detection params
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]
       
        self.C = len(self.classes) # number of classes
        # offset for box center (top left point of each cell)
        self.image_size= cfg.IMAGE_SIZE
        

        self.threshold = 0.2  # confidence scores threshold
        self.iou_threshold = 0.5
        self.model_path = 'model'
        self.sess = tf.Session()
        self.net = Finetuning()
        self.predicts = self.net.forward
        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        
        self.ckpt_file = os.path.join(self.model_path,'Vgg16')
        gpuConfig = tf.ConfigProto(device_count={'gpu':0})
        #gpuConfig.gpu_options.allow_growth = True
        #gpu_options = tf.GPUOptions()
        #config = tf.ConfigProto(gpu_options=gpuConfig)
        self.sess = tf.Session(config=gpuConfig)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.model_path) #获取checkpoints对象
        if ckpt and ckpt.model_checkpoint_path:##判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练
            print('Restoring weights from: ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)#恢复保存的神经网络结构，实现断点续训
        

    def detect_from_file(self, image_file, imshow=True, deteted_boxes_file="boxes.txt",
                     detected_image_file="detected_image.jpg"):
        """Do detection given a image file"""
        # read image
        image = cv2.imread(image_file)
        
        img_h, img_w, _ = image.shape
        predicts = self._detect_from_image(image)
        for i in range(len(predicts)):
            if predicts[i]>=0.3:
                print(self.classes[i]),
                print(predicts[i])
    
        #self.show_results(image, predict_boxes, imshow, deteted_boxes_file, detected_image_file)

    def _detect_from_image(self, image):
        """Do detection given a cv image"""
        img_resized = cv2.resize(image, (self.image_size, self.image_size))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) #opencv习惯使用BGR，将其转换为RGB
        img_resized_np = np.asarray(img_RGB)  #将结构数据转化为ndarray numpy数组
        _images = np.zeros((1, self.image_size, self.image_size, 3), dtype=np.float32)
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        
        predicts = self.sess.run(self.predicts, feed_dict={self.net.images: _images})[0]
        return predicts

if __name__ == "__main__":
    tf.reset_default_graph()
    test = test()
    test.detect_from_file("000017.jpg")