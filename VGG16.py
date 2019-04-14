#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:16:21 2018

@author: zzq
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:36:34 2018

@author: zzq
"""

import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
VGG_MEAN =[103.939,116.799,123.68]  #样本RGB的平均值

class Vgg16():
    def __init__(self  , vgg16__path = None):
        if vgg16__path is None:
            vgg16_path = os.path.join(os.getcwd(),u"vgg16.npy")  #os.getcwd() 方法返回当前工作目录
            print(vgg16_path)
            self.data_dict = np.load(vgg16_path , encoding = u'latin1').item() #遍历其内键值對，导入参数模型
        
        for x in self.data_dict:
            print (x)
            
    def forward(self, images):
        #plt.figure('process picture')
        print("build model started")
        
        start_time = time.time() #获取向前传播的开始时间
        
        rgb_scaled = images * 255.0 #组像素乘以255.0
        
        #从GRB转换色彩通道到BGR 也可以使用cv中的 CRBtoBRG
        red , green ,blue = tf.split(rgb_scaled,3,3)
        assert red.get_shape().as_list()[1:]==[224,224,1]
        assert green.get_shape().as_list()[1:] ==[224,224,1]
        assert blue.get_shape().as_list()[1:] ==[224,224,1]
        bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1],red - VGG_MEAN[2]],3)
        assert bgr.get_shape().as_list()[1:]==[224,224,3]
        
        # 接下来构建VGG的16層网络（5段卷积 3层全连接）逐层根据命名空间读取网络参数
        # 第一断卷积 两个卷积层 后面接最大池化层 用来缩小图片尺寸
        self.conv1_1 = self.conv_layer(bgr,'conv1_1')
        #传入命名空间的name 来获取该层的卷积核和偏置，并做卷积运算，最后返回经过激活函数后的值
        self.conv1_2 = self.conv_layer(self.conv1_1,'conv1_2')
        #根据传入的pooling名字对该层做相应的池化操作
        self.pool1 = self.max_pool_2x2(self.conv1_2,'pool1')
        #下面向前传播过程与第一段相同
        self.conv2_1 = self.conv_layer(self.pool1,'conv2_1')
        #传入命名空间的name 来获取该层的卷积核和偏置，并做卷积运算，最后返回经过激活函数后的值
        self.conv2_2 = self.conv_layer(self.conv2_1,'conv2_2')
        #根据传入的pooling名字对该层做相应的池化操作
        self.pool2 = self.max_pool_2x2(self.conv2_2,'pool2')
        
        #第三段卷积，包含三个卷积层，一个最大池化层
        self.conv3_1 = self.conv_layer(self.pool2,'conv3_1')
        #传入命名空间的name 来获取该层的卷积核和偏置，并做卷积运算，最后返回经过激活函数后的值
        self.conv3_2 = self.conv_layer(self.conv3_1,'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2,'conv3_3')
        #根据传入的pooling名字对该层做相应的池化操作
        self.pool3 = self.max_pool_2x2(self.conv3_3,'pool3')
        
        #第四段卷积，包含三个卷积层，一个最大池化层
        self.conv4_1 = self.conv_layer(self.pool3,'conv4_1')
        #传入命名空间的name 来获取该层的卷积核和偏置，并做卷积运算，最后返回经过激活函数后的值
        self.conv4_2 = self.conv_layer(self.conv4_1,'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2,'conv4_3')
        #根据传入的pooling名字对该层做相应的池化操作
        self.pool4 = self.max_pool_2x2(self.conv4_3,'pool4')
        
        #第五层卷积，包含三个卷积层，一个最大池化层
        self.conv5_1 = self.conv_layer(self.pool4,'conv5_1')
        #传入命名空间的name 来获取该层的卷积核和偏置，并做卷积运算，最后返回经过激活函数后的值
        self.conv5_2 = self.conv_layer(self.conv5_1,'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2,'conv5_3')
        #根据传入的pooling名字对该层做相应的池化操作
        self.pool5 = self.max_pool_2x2(self.conv5_3,'pool5')
        
        # 第6层全连接层
        
        
        #第八层全连接
        
        # 经过最后一层的全连接 再做softmax分类 得到属于给类别的概率
        
        end_time = time.time() #得到向前传播的结束时间
        print(("time consuming : %f" %(end_time-start_time)))
        self.data_dict = None # 清空本次读取到的模型参数字典
        return self.pool5
    #定义卷积运算
    def conv_layer(self,x,name):
        with tf.variable_scope(name):
            w = tf.Variable(self.get_conv_filter(name)) #根据命名空间找到对应卷积层的网络参数
            conv = tf.nn.conv2d(x,w,[1,1,1,1],padding = 'SAME') #卷积计算
            conv_biases = tf.Variable(self.get_bias(name)) #读到偏置
            result = tf.nn.relu(tf.nn.bias_add(conv,conv_biases))  #加上偏置，并做激活运算
            return result
    
    # 定义获取卷积核的函数
    def get_conv_filter(self , name):
        #根据命名空间name从参数字典中取得对应的卷积核
        return self.data_dict[name][0]
    
    def get_bias(self,name):
        # 根据命名空间name 从参数字典中取到对应的卷积核
        return self.data_dict[name][1]
    
    # 定义最大池化操作
    
    def max_pool_2x2(self ,x ,name):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'SAME',name = name)
    
    #定义全连接层的前向传播计算
    
    def fc_layer(self,x,name):
        with tf.variable_scope(name):               #根据命名空间name做全连接层的计算
            shape = x.get_shape().as_list()          #获取该层的纬度信息列表
            
            print('fc_layer shape',shape)
            dim = 1
            for i in shape[1:]: 
             #将每层的纬度相乘
                dim*=i
           #改变特征图的形状，也就是将得到的多维特征做拉伸操作，只载进入第六层全连接层做该操作
            x = tf.reshape(x,[-1,dim])
            w = tf.Variable(self.get_fc_weight(name)) #读到权重值
            b = tf.Variable(self.get_bias(name))    #读到偏置值
           
            result = tf.nn.bias_add(tf.matmul(x,w),b)
            return result
        #定义获取权重的函数
    def get_fc_weight(self,name):
        return self.data_dict[name][0]
      