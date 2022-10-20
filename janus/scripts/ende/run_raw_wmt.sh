#!/bin/bash
src=ro
tgt=en

export CUDA_VISIBLE_DEVICES=0,1,2,3

# data_dir=/data1/lxb/wmt14_ende/wmt_test/data-bin
# data_dir=/data1/lxb/wmt16.en-de.disco.dist/
# save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_raw_max

# data_dir=/data1/lxb/wmt14_cmlm/wmt16.en-ro.dist/data-bin/
# save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_enro_cmlm

data_dir=/data1/lxb/wmt14_cmlm/wmt16.en-ro.raw/data-bin/
# save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_enro_cmlm_raw
save_dir=/data1/lxb/nmt_checkpoint/janus/baseline_roen

python -m torch.distributed.launch --nproc_per_node=4 --master_port=1230 \
    $(which fairseq-train)  $data_dir \
    --ddp-backend=no_c10d \
    --distributed-backend 'nccl' \
    --distributed-no-spawn \
    --task translation_lev \
    --arch cmlm_transformer \
    --noise random_mask \
    -s $src -t $tgt \
    --share-all-embeddings \
    --dropout 0.3 \
    --criterion nat_loss \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 \
    --max-tokens 8192 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --lr-scheduler inverse_sqrt --weight-decay 0.01 \
    --no-progress-bar \
    --seed 64 \
    --fp16 \
    --max-update 200000 --warmup-updates 10000 --warmup-init-lr 1e-07 \
    --save-dir $save_dir | tee -a $save_dir/train.log \