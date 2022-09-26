PRE_TRAIN_DIR=checkpoints/baseline_model
SAVE_DIR=checkpoints/DITTO_openended
mkdir -p $SAVE_DIR


python -u ./train.py \
--task language_modeling_pe_rep datas/data-bin/wikitext-103 \
    --user-dir ./fairseq/custom --arch transformer_lm_ul --max-tokens 1536 --tokens-per-sample 1536 \
    --fp16 --max-update 10000 --max-lr 1.0e-2 --t-mult 2 --lr-period-updates 270000 \
    --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 0 --warmup-init-lr 1e-07 --min-lr 1e-09 \
    --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 6 --seed 1 --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 100 \
    --keep-interval-updates 2 --no-progress-bar --log-interval 10 \
    --rank-alpha 1.0 --sequence-level-train-rate 0.5 \
    --rep_reduce_gamma 0.5 \
    --reset-lr-scheduler --reset-optimizer --reset-meters \
    --compute-metrics-interval 1 \
    --restore-file $PRE_TRAIN_DIR/checkpoint_best.pt \
    --criterion cross_entropy_wcustom_metrics \
    --save-dir $SAVE_DIR \
    --tensorboard-logdir $SAVE_DIR 2>&1 | tee -a $SAVE_DIR/log.txt