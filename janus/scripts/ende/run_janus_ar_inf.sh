#!/bin/bash
src=en
tgt=de
# src=de
# tgt=en

export CUDA_VISIBLE_DEVICES=0

root_dir=/opt/data/private/data/nmt_data
data_dir=$root_dir/wmt14_ende/data-bin
save_dir=$root_dir/ckpt/


fairseq-generate $data_dir \
    --user-dir janus/src \
    --task janus_translation \
    -s $src -t $tgt \
    --inference-mode 'ar' \
    --path $save_dir/checkpoint_last.pt \
    --batch-size 64 --beam 5 --remove-bpe  > $save_dir/log.iter10.txt

bash scripts/compound_split_bleu.sh $save_dir/log.iter10.txt
