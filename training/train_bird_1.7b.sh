set -e

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,2"

LR=1.5e-5
EPOCHS=5
CONFIG_FILE="./accelerate_config_1.7b.yaml"
PER_DEVICE_TRAIN_BATCH_SIZE=2
MODEL_PATH="/data/home/vkropoti/models/qwen3-1.7B-after_train"
CKPT_NUM=15
BASE_NAME="qwen3_1.7b_bird_lr${LR}_epochs${EPOCHS}"
CKPT_DIR="/data/home/vkropoti/models/ckpts_qwen3-1.7b-bird/$BASE_NAME"
LOG_DIR="/data/home/vkropoti/sql_data/train/train_logs/$BASE_NAME"
DATASET_DIR="/data/home/vkropoti/sql_data/train/train_dataset_sft_v2"

accelerate launch \
--main_process_port 10000 \
--config_file $CONFIG_FILE \
train.py \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --block_size 7500 \
    --seed 42 \
    --pretrained_model_name_or_path $MODEL_PATH \
    --epochs $EPOCHS \
    --lr $LR \
    --ckpt_num $CKPT_NUM \
    --tensorboard_log_dir $LOG_DIR \
    --output_ckpt_dir $CKPT_DIR \
    --sft_data_dir $DATASET_DIR \
    --mode sft