#!/bin/bash
src=en
tgt=de
# src=de
# tgt=en

export CUDA_VISIBLE_DEVICES=0

data_dir=/data1/lxb/wmt14_ende/data-bin
save_dir=/data1/lxb/nmt_checkpoint/janus/janus_distill_ende_kl13_new

fairseq-generate $data_dir \
    --user-dir examples/janus/src \
    --task janus_translation \
    -s $src -t $tgt \
    --inference-mode 'ar' \
    --path $save_dir/ave.pt \
    --batch-size 64 --beam 5 --remove-bpe  > $save_dir/ar_gen.txt

bash scripts/compound_split_bleu.sh $save_dir/ar_gen.txt
