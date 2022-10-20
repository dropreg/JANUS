#!/bin/bash
src=de
tgt=en

export CUDA_VISIBLE_DEVICES=0,1

data_dir=/opt/data/private/data/iwslt14.tokenized.de-en/data-bin/
save_dir=/opt/data/private/ckpt/janus/distill_baseline_seed2048

# --restore-file $restore_file \
# --reset-optimizer --reset-dataloader --reset-meters \
fairseq-train $data_dir \
    --user-dir examples/janus/src \
    --task janus_translation \
    --arch janus_transformer_iwslt_de_en \
    --share-all-embeddings \
    --dropout 0.3 \
    --criterion janus_distill_loss \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 -s $src -t $tgt \
    --max-tokens 4096 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --no-progress-bar \
    --seed 2048 \
    --fp16 \
    --max-update 200000 --warmup-updates 10000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
    --save-dir $save_dir | tee -a $save_dir/train.log \
