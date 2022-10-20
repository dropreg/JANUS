#!/bin/bash
src=de
tgt=en

export CUDA_VISIBLE_DEVICES=6

data_dir=/data1/lxb/dataset/small_nmt_data/iwslt_data/de-en_file/databin
save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_distill_raw_kl33

python scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/ave.pt \
                    --num-epoch-checkpoints 5 --checkpoint-upper-bound 250

fairseq-generate $data_dir \
    --user-dir examples/janus/src \
    --task janus_translation \
    -s $src -t $tgt \
    --inference-mode 'ar' \
    --path $save_dir/ave.pt \
    --batch-size 128 --beam 5 --remove-bpe --quiet \
