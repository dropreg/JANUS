#!/bin/bash
# src=en
# tgt=de
src=en
tgt=de

export CUDA_VISIBLE_DEVICES=0

root_dir=/opt/data/private/data/nmt_data
data_dir=$root_dir/wmt14_ende/data-bin
save_dir=$root_dir/ckpt/

# python scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/ave.pt \
#                     --num-epoch-checkpoints 5 --checkpoint-upper-bound 90

fairseq-generate $data_dir \
    --user-dir janus/src \
    --task janus_translation \
    -s $src -t $tgt \
    --inference-mode 'nar' \
    --gen-subset test \
    --path $save_dir/checkpoint_last.pt \
    --iter-decode-max-iter 10 \
    --iter-decode-with-beam 3 --remove-bpe \
    --iter-decode-force-max-iter \
    --batch-size 8 > $save_dir/log.iter10.txt

bash scripts/compound_split_bleu.sh $save_dir/log.iter10.txt
