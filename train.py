#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:13:37 2018

@author: zzq
"""

import os
import argparse
import datetime
import tensorflow as tf
import config as cfg
from utils.timer import Timer
from utils.pascal_voc import pascal_voc
from Finetuning import Finetuning
slim = tf.contrib.slim

class Solver():
    def __init__(self, net, data,is_training=True):
        self.net = net
        self.data = data
        self.image_size= cfg.IMAGE_SIZE
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.model_path  = '/home/zzq/test/VGG16/model'
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()
         
        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.model_path,'Vgg16')
        
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)
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
        
        self.writer.add_graph(self.sess.graph)
        
    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter ) == 0:

                    train_timer.tic()
                    summary_str, loss,_ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
                    print(step),
                    print(loss),
                   
                    print(train_timer.average_time)
                    #print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)
def main():
    tf.reset_default_graph() 
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    net = Finetuning()
    pascal = pascal_voc('train')
    
    solver = Solver(net, pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()