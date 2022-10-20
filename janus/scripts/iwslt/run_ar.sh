#!/bin/bash
src=de
tgt=en

export CUDA_VISIBLE_DEVICES=6

data_dir=/data1/lxb/dataset/small_nmt_data/iwslt_data/de-en_file/databin
save_dir=/data1/lxb/nmt_checkpoint/janus/ar_baseline

fairseq-train $data_dir \
    --user-dir examples/janus/src \
    --task janus_translation \
    --arch janus_transformer_iwslt_de_en \
    --share-all-embeddings \
    --dropout 0.3 \
    --criterion ar_loss \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 -s $src -t $tgt \
    --max-tokens 8192 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --no-progress-bar \
    --seed 64 \
    --fp16 \
    --max-update 100000 --warmup-updates 10000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
    --save-dir $save_dir | tee -a $save_dir/train.log \