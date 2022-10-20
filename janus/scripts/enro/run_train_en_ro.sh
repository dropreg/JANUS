#!/bin/bash
src=ro
tgt=en

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_dir=/data1/lxb/wmt14_cmlm/wmt16.en-ro.raw/data-bin/
save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_roen_janus_kl

python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 \
    $(which fairseq-train)  $data_dir \
    --ddp-backend=no_c10d \
    --distributed-backend 'nccl' \
    --distributed-no-spawn \
    --user-dir examples/janus/src \
    --task janus_translation \
    --arch janus_transformer \
    -s $src -t $tgt \
    --share-all-embeddings \
    --dropout 0.3 \
    --criterion janus_loss \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 \
    --max-tokens 4096 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --lr-scheduler inverse_sqrt --weight-decay 0.01 \
    --no-progress-bar \
    --seed 64 \
    --fp16 \
    --max-update 200000 --warmup-updates 10000 --warmup-init-lr 1e-07 \
    --save-dir $save_dir | tee -a $save_dir/train.log \