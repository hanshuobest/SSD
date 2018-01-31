#!/bin/bash
# This is the eval script.

DATASET_DIR=/home/han/DataSets/VOCtest_06-Nov-2007/VOC2007_tfrecord/
EVAL_DIR=/home/han/SSD/logs/    # Directory where the results are saved to
CHECKPOINT_PATH=../VGG_VOC0712_SSD_300x300_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_iter_120000.ckpt

#dataset_name这个参数在代码里面写死了
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=8
