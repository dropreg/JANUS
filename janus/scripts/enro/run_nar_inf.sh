#!/bin/bash
src=ro
tgt=en
export CUDA_VISIBLE_DEVICES=0
data_dir=/data1/lxb/wmt16.en-ro.raw/data-bin/
save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_roen_consistent

# src=en
# tgt=ro
# export CUDA_VISIBLE_DEVICES=0
# data_dir=/data1/lxb/wmt16.en-ro.raw/data-bin/
# save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_enro_nokl

python scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/ave.pt \
                    --num-epoch-checkpoints 5 --checkpoint-upper-bound 100

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
