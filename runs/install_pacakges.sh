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
pip install huggingface-hub==0.4.0