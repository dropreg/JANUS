#!/bin/bash
src=de
tgt=en

export CUDA_VISIBLE_DEVICES=0

root_dir=/opt/data/private/data/nmt_data
data_dir=$root_dir/iwslt14.tokenized.de-en/data-bin
save_dir=$root_dir/ckpt/

# python scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/ave.pt \
#                     --num-epoch-checkpoints 5 --checkpoint-upper-bound 175

fairseq-generate $data_dir \
    --user-dir janus/src \
    --task janus_translation \
    -s $src -t $tgt \
    --gen-subset test \
    --inference-mode 'nar' \
    --path $save_dir/checkpoint_last.pt \
    --iter-decode-max-iter 10 \
    --iter-decode-with-beam 3 --remove-bpe \
    --iter-decode-force-max-iter \
    --batch-size 128 > $save_dir/log.iter10.txt

bash scripts/compound_split_bleu.sh $save_dir/log.iter10.txt
