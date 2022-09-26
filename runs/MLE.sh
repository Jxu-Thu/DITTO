
SAVE_DIR=checkpoints/baseline_model
mkdir -p $SAVE_DIR

python -u train.py \
--task language_modeling_with_generation datas/data-bin/wikitext-103 \
    --user-dir ./fairseq/custom --arch transformer_lm_ul --max-tokens 1536 --tokens-per-sample 1536 \
    --fp16 --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 \
    --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
    --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 10000 \
    --keep-interval-updates 2 --no-progress-bar --log-interval 100 \
    --criterion cross_entropy_wcustom_metrics \
    --save-dir $SAVE_DIR \
    --tensorboard-logdir $SAVE_DIR 2>&1 | tee -a $SAVE_DIR/log.txt

