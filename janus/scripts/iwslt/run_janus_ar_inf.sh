#!/bin/bash
src=de
tgt=en

export CUDA_VISIBLE_DEVICES=0

root_dir=/opt/data/private/data/nmt_data
data_dir=$root_dir/iwslt14.tokenized.de-en/data-bin
save_dir=$root_dir/ckpt/

# python scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/ave.pt \
#                     --num-epoch-checkpoints 5 --checkpoint-upper-bound 250

fairseq-generate $data_dir \
    --user-dir janus/src \
    --task janus_translation \
    -s $src -t $tgt \
    --inference-mode 'ar' \
    --path $save_dir/checkpoint_last.pt \
    --batch-size 128 --beam 5 --remove-bpe --quiet \
