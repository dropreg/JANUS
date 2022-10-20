#!/bin/bash
src=en
tgt=ro

export CUDA_VISIBLE_DEVICES=1

# data_dir=/data1/lxb/small_nmt_data/iwslt_data/de-en_file/databin
# save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_armlr

data_dir=/data1/lxb/large_nmt_data/wmt16.en-ro.raw/data-bin/
save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_en_ro_kl

fairseq-generate $data_dir \
    --user-dir examples/janus/src \
    --task janus_translation \
    --inference-mode 'ar' \
    --path $save_dir/checkpoint50.pt \
    --beam 5 --remove-bpe  > $save_dir/ar_gen.txt

bash scripts/compound_split_bleu.sh $save_dir/ar_gen.txt