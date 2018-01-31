#!/bin/bash

#The directory where the dataset files are stored.
DATASET_DIR=/home/han/DataSets/VOCtest_06-Nov-2007/VOC2007_tfrecord/
#../../../../common/dataset/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007_tfrecord/

#Directory where checkpoints and event logs are written to.
TRAIN_DIR=/home/han/SSD/logs/

#The path to a checkpoint from which to fine-tune
CHECKPOINT_PATH=../VGG_VOC0712_SSD_300x300_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_iter_120000.ckpt
#../../../../common/models/tfmodlels/SSD/VGG_VOC0712_SSD_300x300_ft_iter_120000/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt



python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=16
