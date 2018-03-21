＃filename = tf_convert_data.sh
#!/bin/bash
#This is a shell script to convert Pascal VOC datasets(2007 and 2012) into TF-Records only.
# 这是一个将VOC datasets（2007和2012）转成TF-Records的脚本

#Directory where the original dataset is stored
DATASET_DIR=/home/han/DataSets/VOCdevkit-2012/VOC2012/

#Output directory where to store TFRecords files
OUTPUT_DIR=/home/han/DataSets/VOCdevkit-2012/VOC2012_tfrecord/

python ./tf_convert_data.py \
    --dataset_name=VOC2012 \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2012_train \
    --output_dir=${OUTPUT_DIR}
