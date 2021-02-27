# Topic Segmentation based on Graph Neural Networks
This repository contains the code and dataset for papar 《Topic Segmentation based on Graph Neural Networks》,implemented in Pytorch.
Please note that our work is mainly based on [《Text Segmentation as a Supervised Learning Task》](https://arxiv.org/abs/1803.09337) and the original code for their work is on [text-segmentation](https://github.com/koomri/text-segmentation).
In addition, the idea of our method is basically from [《Every Document Owns Its Structure: Inductive Text Classification via
Graph Neural Networks》](https://arxiv.org/abs/2004.13826)([code for TextING](https://github.com/CRIPAC-DIG/TextING)) and [《Graph Convolutional Networks for Text Classification》](https://arxiv.org/abs/1809.05679)([code for TextGCN](https://github.com/yao8839836/text_gcn)).
Thank them all for their works.

# Requirements
pytorch=1.7.1

# Required Resources
word2vec:
https://drive.google.com/a/audioburst.com/uc?export=download&confirm=zrin&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

wiki-727K, wiki-50:
https://www.dropbox.com/sh/k3jh0fjbyr0gw0a/AADzAd9SDTrBnvs1qLCJY5cza?dl=0

WIKI-SECTION:
https://github.com/sebastianarnold/WikiSection

CHOI:
https://github.com/koomri/text-segmentation/tree/master/data/choi

CLINICAL:
https://github.com/pinkeshbadjatiya/neuralTextSegmentation/tree/master/code/data/clinical

CITIES,ELEMENTS:
http://groups.csail.mit.edu/rbg/code/mallows/

MANIFESTO:
https://github.com/koomri/text-segmentation/tree/master/data/manifesto

word2vec:
https://drive.google.com/a/audioburst.com/uc?export=download&confirm=zrin&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM


# Usage
Start training as:
```python
python run.py [--train_data_type {wiki,choi,wikisection}]
              [--etype {one_hot,glove,randn,w2v,bert}] [--encoder_fine_tune]
              [--sr_choose {f_model,l_model,s_model,t_model,g_model,b_model,random_baseline}]
              [--tr_choose {balanced,left,right}] [--use_leaf_rnn]
              [--gr_choose {textgcn,texting,ING_GCN,S_ING_GCN}]
              [--gnn_window_size GNN_WINDOW_SIZE]
              [--texting_gru_step TEXTING_GRU_STEP]
```
Example:
```python
python run.py --train_data_type=wiki --etype=w2v --sr_choose=g_model --gr_choose=ING_GCN
```
