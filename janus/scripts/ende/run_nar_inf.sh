#!/bin/bash
# src=en
# tgt=de
src=en
tgt=de

export CUDA_VISIBLE_DEVICES=0

# data_dir=/data1/lxb/en_de_rawdata
# data_dir=/data1/lxb/wmt16.en-de.raw/data-bin
data_dir=/data1/lxb/wmt14_ende/data-bin
save_dir=/data1/lxb/nmt_checkpoint/janus/janus_distill_ende_kl13_new
# save_dir=/data1/lxb/nmt_checkpoint/janus/janus_distill_deen_kl13_new

python scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/ave.pt \
                    --num-epoch-checkpoints 5 --checkpoint-upper-bound 90


fairseq-generate $data_dir \
    --user-dir examples/janus/src \
    --task janus_translation \
    -s $src -t $tgt \
    --inference-mode 'nar' \
    --gen-subset test \
    --path $save_dir/ave.pt \
    --iter-decode-max-iter 10 \
    --iter-decode-with-beam 3 --remove-bpe \
    --iter-decode-force-max-iter \
    --batch-size 8 > $save_dir/gen.txt

bash scripts/compound_split_bleu.sh $save_dir/gen.txt
