
# training bart model with 22g BERT data
DATA_PATH=/data1/lxb/huawei_pretrain/bert_data_22g/train/data-bin
# DATA_PATH=/data1/lxb/huawei_pretrain/gpt_160g/
MODEL_PATH=/data1/lxb/huawei_pretrain/pretraining/ckpt/janus/kl_22g
BART_PATH=/data1/lxb/huawei_pretrain/pretraining/bart.base/model.pt

TOTAL_NUM_UPDATE=1000000
WARM_UPDATE=10000

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 \
    $(which fairseq-train) $DATA_PATH \
    --restore-file $BART_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --user-dir examples/janus_pretrain/src \
    --arch janus_bart_base \
    --task janus_denoising \
    --replace-length 1 \
    --mask 0.3 \
    --mask-length span-poisson \
    --criterion janus_pretrain_loss \
    --sample-break-mode complete \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 0.0001 \
    --warmup-updates $WARM_UPDATE --total-num-update $TOTAL_NUM_UPDATE --max-update $TOTAL_NUM_UPDATE \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences 3 --update-freq 16 \
    --log-format simple \
    --log-interval 100 \
    --ddp-backend=no_c10d \
    --distributed-backend 'nccl' \
    --distributed-no-spawn \
    --tensorboard-logdir $MODEL_PATH \
    --save-dir $MODEL_PATH \
    --fp16 \
    --save-interval-updates 20000 \
    --skip-invalid-size-inputs-valid-test \
    2>&1 | tee -a ${MODEL_PATH}/train.log