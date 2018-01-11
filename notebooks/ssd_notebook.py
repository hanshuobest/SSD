# -*- coding: utf-8 -*-

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import time
from datetime import timedelta
from datetime import datetime

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
# from notebooks import visualization
import visualization
import multiprocessing
import threading
import glob
from multiprocessing import JoinableQueue



class Predict(object):
    def __init__(self , dirpath):
        # print("dirpath:" , dirpath)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.net_shape = (300, 300)
            self.data_format = "NHWC"
            self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
            self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                    self.img_input, None, None, self.net_shape, self.data_format,
                    resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(self.image_pre, 0)
            self.reuse = True if 'ssd_net' in locals() else None
            self.ssd_net = ssd_vgg_300.SSDNet()

            with slim.arg_scope(ssd_net.arg_scope(data_format=self.data_format)):
                self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False,reuse=False)
            self.saver = tf.train.Saver()

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.08)
        self.config = tf.ConfigProto(gpu_options=self.gpu_options)
        self.sess = tf.Session(graph=self.graph , config=self.config)

        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess ,"../checkpoints/ssd_300_vgg.ckpt")


    def predict(self , img, select_threshold=0.5, nms_threshold=.45):


        rimg, rpredictions, rlocalisations, rbbox_img = self.sess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                 feed_dict={self.img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)


        return rclasses, rscores, rbboxes

class Predict2(object):
    def __init__(self , dirpath):
        print("dirpath:" , dirpath)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.net_shape = (300, 300)
            self.data_format = "NHWC"
            self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
            self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                    self.img_input, None, None, self.net_shape, self.data_format,
                    resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(self.image_pre, 0)
            self.reuse = True if 'ssd_net' in locals() else None
            self.ssd_net = ssd_vgg_300.SSDNet()

            with slim.arg_scope(ssd_net.arg_scope(data_format=self.data_format)):
                self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False,reuse=False)
            self.saver = tf.train.Saver()

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.config = tf.ConfigProto(gpu_options=self.gpu_options)
        self.sess = tf.Session(graph=self.graph , config=self.config)

        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess ,"../checkpoints/ssd_300_vgg.ckpt")


    def predict(self , img, select_threshold=0.5, nms_threshold=.45):
        start_time = time.time()

        rimg, rpredictions, rlocalisations, rbbox_img = self.sess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                 feed_dict={self.img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

        total_time = time.time() - start_time
        print("cost time:", total_time)
        return rclasses, rscores, rbboxes

class Predict3(object):
    def __init__(self , dirpath):
        print("dirpath:" , dirpath)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.net_shape = (300, 300)
            self.data_format = "NHWC"
            self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
            self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                    self.img_input, None, None, self.net_shape, self.data_format,
                    resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(self.image_pre, 0)
            self.reuse = True if 'ssd_net' in locals() else None
            self.ssd_net = ssd_vgg_300.SSDNet()

            with slim.arg_scope(ssd_net.arg_scope(data_format=self.data_format)):
                self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False,reuse=False)
            self.saver = tf.train.Saver()

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.config = tf.ConfigProto(gpu_options=self.gpu_options)
        self.sess = tf.Session(graph=self.graph , config=self.config)

        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess ,"../checkpoints/ssd_300_vgg.ckpt")


    def predict(self , img, select_threshold=0.5, nms_threshold=.45):
        start_time = time.time()

        rimg, rpredictions, rlocalisations, rbbox_img = self.sess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                 feed_dict={self.img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

        total_time = time.time() - start_time
        print("cost time:", total_time)
        return rclasses, rscores, rbboxes


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
# gpu_options = tf.GPUOptions(allow_growth=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# isess = tf.InteractiveSession(config=config)
# isess2 = tf.InteractiveSession(config = config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()

with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
# isess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(isess, ckpt_filename)
#
# isess2.run(tf.global_variables_initializer())
# saver2 = tf.train.Saver()
# saver2.restore(isess2 , ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# Main image processing routine.
def process_image2(sess , img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    start_time = time.time()

    rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

    total_time = time.time() - start_time
    print("cost time:" , total_time)
    return rclasses, rscores, rbboxes

def worker(index , images):
    print("index:" , index)
    print("start working...")
    start_time = time.time()

    p = Predict(None)
    num = len(images)
    for i in range(num):
        img = mpimg.imread(images[i])
        p.predict(img)

    print("end working...")
    total_time = time.time() - start_time
    print("cost time:", total_time)
    print("pid:" , os.getpid())

def usb_test(images):
    print("start working...")
    # start_time = time.time()
    num = len(images)
    p = Predict(None)
    for i in range(num):
        img = mpimg.imread(images[i])
        p.predict(img)
    print("end working...")

    # total_time = time.time() - start_time
    # print("cost time:", total_time)
    print("pid:" , os.getpid())


# img = mpimg.imread(path + image_names[-5])
# img = mpimg.imread("../demo/timg.jpeg")
# rclasses, rscores, rbboxes =  process_image2(isess , img)
# rclasses2, rscores2, rbboxes2 =  process_image2(isess2 , img)

# visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
# visualization.plt_bboxes(img, rclasses2, rscores2, rbboxes2)


imgs_dir = "/home/han/Faster-R-CNN/data/VOCdevkit2007/VOC2007/JPEGImages"
img_lists = glob.glob(imgs_dir + "/*.jpg")
print(len(img_lists))

start_time = datetime.now()
# 进程测试
p_array = []
for i in range(8):
    p_i = multiprocessing.Process(target=worker , args=(i , img_lists[600 * i:(i + 1) * 600]))
    p_array.append(p_i)
for i in p_array:
    i.start()
for i in p_array:
    i.join()

# 串行测试
# usb_test(img_lists)

# start_time = time.time()
# for i in range(8):
#     usb_test(img_lists[600 * i:(i + 1) * 600])
# cost_time = time.time() - start_time
# print("cost total time:" , cost_time)



# 线程测试
# threads = []
# t1 = threading.Thread(target=worker, args=(1 , img))
# t2 = threading.Thread(target=worker , args=(2 , img))
# threads.append(t1)
# threads.append(t2)
#
# for t in threads:
#     # t.setDaemon(True)
#     t.start()
# start = datetime.now()
# threads = []
# for i in range(8):
#     t = threading.Thread(target=worker , args=(i , img_lists[600 * i : 600 * (i + 1)]))
#     threads.append(t)
# for t in threads:
#     t.start()
# for t in threads:
#     t.join()
cost_time = datetime.now() - start_time
print("all cost time :%s" % timedelta.total_seconds(cost_time))
