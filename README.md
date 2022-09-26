# DITTO

The codes can be used to reproduce the results [DITTO](https://arxiv.org/abs/2206.02369) for open-ended generation (Wikitext-103) and abstractive summrization (DNN/DailyMail).

[Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation](https://arxiv.org/abs/2206.02369), by [Jin Xu](https://jxu-thu.github.io/), Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li and [Jian Li](https://people.iiis.tsinghua.edu.cn/~jianli/), is a research work for analyzing the sentence-level repetition issues, mitigating it and improving the learning ability and generalizability of language models at the training stage.

The codes for open-ended generations include
* `runs/MLE.sh`: Normal MLE training
* `runs/DITTO.sh`: Fine-tune the MLE trained models with DITTO

All configuration and architecture have been included in the .sh scripts.

### Installation

```
conda create --name py37_torch python=3.7
pip install torch==1.4.0
pip install --editable .
pip install nltk
pip install pandas
pip install ipython
pip install pytorch-transformers   # (optional); for GPT-2 fine-tuning
pip install tensorflow==1.14
pip install tensorboardX           # (optional); for tensorboard logs
pip install --user networkx==2.3
pip install --user matplotlib==3.1.0
pip install seaborn
pip install mauve-text==0.2.0
pip install transformers==4.17.0
pip install huggingface-hub==0.4.0conda create --name py37_torch python=3.7
pip install torch==1.4.0
pip install --editable .
pip install nltk
pip install pandas
pip install ipython
pip install pytorch-transformers   # (optional); for GPT-2 fine-tuning
pip install tensorflow==1.14
pip install tensorboardX           # (optional); for tensorboard logs
pip install --user networkx==2.3
pip install --user matplotlib==3.1.0
pip install seaborn
pip install mauve-text==0.2.0
pip install transformers==4.17.0
pip install huggingface-hub==0.4.0
```

### Dataset
Prepare the dataset following [Unlikelihood](https://github.com/facebookresearch/unlikelihood_training) by downloading and unpacking the binarized wikitext-103 dataset (160MB, install wget if needed):
```
mkdir datas && cd datas
wget https://dl.fbaipublicfiles.com/unlikelihood/wikitext-103_v0.tar.gz
tar xzvf wikitext-103_v0.tar.gz
cd ..
```

### Create a checkpoint folder
```
mkdir checkpoints
```

### MLE Training
We train the models using 8*Tesla V100 32GB gpu(s).
```
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
    --max-epoch 30 \
    --tensorboard-logdir $SAVE_DIR 2>&1 | tee -a $SAVE_DIR/log.txt
 ```

### DITTO Fine-tuning
We further fine-tune the pre-trained models with DITTO loss. We train the models using 8*Tesla V100 32GB gpu(s).
```
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
```

### Evaluation
Run the below commands for greedy decoding
```
TASK=DITTO_openended
SAVE_LOG_PATH=evaluation_logs

mkdir -p $SAVE_LOG_PATH

DS_SPLIT=test
bms=1
bnb=0
tpk=1
tpp=0
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

# Note that mauve calculation is very slow.
python report_metrics.py \
    --eval-dir $SAVE_LOG_PATH \
    --report-mauve \
    --model-names $TASK
```

Run the below commands for top-p decoding
```
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
```

### Analyze Self-reinforcement Effect
We prepare some sentences to construct repetitive sequences for analyses. We provide some sentences in data/wiki_sentences.txt and you can use any other sentences for analyses.

You can change the $SAVE_PATH to the baseline (MLE) or DITTO ckpt to see the comparison of the self-reinforcement effect.
```
bms=1
bnb=0
tpk=1
tpp=0
sttpk=1
sttpp=0
TASK=DITTO_openended
SAVE_PATH=checkpoints/$TASK

DOC_FILE=data/wiki_sentences.txt
EVAL_SAVE_DIR=analyses/$TASK
FINAL_DIR=analyses_final/$TASK
ls -l $SAVE_PATH

mkdir -p $EVAL_SAVE_DIR

python -u fairseq/custom/eval_manual_rep_document.py \
    --batch-size-single-prediction 1536 --batch-size-completion 48 \
    --data-prefix-length 50 --completion-length 100 \
    --ckpt best \
    --model-path $SAVE_PATH \
    --save_dir $EVAL_SAVE_DIR \
    --document_path $DOC_FILE \
    --beam-size $bms --beam-ngram-block $bnb --topp $tpp --topk $tpk --singletoken-topk $sttpk --singletoken-topp $sttpp \
    --data-dir datas/data-bin/wikitext-103 \
    --base-dir ./

# Calculate the statictics and draw several figures

mkdir -p $FINAL_DIR
python -u fairseq/custom/repetition_para_parse.py \
    --pkl_dir $EVAL_SAVE_DIR \
    --save_dir $FINAL_DIR 

```

