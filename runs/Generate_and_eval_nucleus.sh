
TASK=DITTO_openended
SAVE_LOG_PATH=evaluation_logs

mkdir -p $SAVE_LOG_PATH

DS_SPLIT=test
bms=1
bnb=0
tpk=1
tpp=0.9
sttpk=1
sttpp=0


SAVE_PATH=checkpoints/$TASK
ls -l $SAVE_PATH
python -u fairseq/custom/evaluation.py \
    --batch-size-single-prediction 1536 --batch-size-completion 48 \
    --data-prefix-length 50 --completion-length 100 \
    --save-path $SAVE_LOG_PATH --ckpt best \
    --model-path $SAVE_PATH \
    --data-split $DS_SPLIT \
    --beam-size $bms --beam-ngram-block $bnb --topp $tpp --topk $tpk --singletoken-topk $sttpk --singletoken-topp $sttpp \
    --data-dir datas/data-bin/wikitext-103 \
    --base-dir ./


python report_metrics.py \
    --eval-dir $SAVE_LOG_PATH \
    --report-mauve \
    --model-names $TASK