# 《基于篇章结构图网络的话题分割》(Topic Segmentation via Discourse Structure Graph Network)
该仓库包含了《基于篇章结构图网络的话题分割》文章的代码和数据集，代码使用pytorch框架实现。
我们的工作主要基于[《Text Segmentation as a Supervised Learning Task》](https://arxiv.org/abs/1803.09337) ，它的源代码是[text-segmentation](https://github.com/koomri/text-segmentation)。
另外，我们的模型灵感主要来自于 [《Every Document Owns Its Structure: Inductive Text Classification via
Graph Neural Networks》](https://arxiv.org/abs/2004.13826)([TextING的代码](https://github.com/CRIPAC-DIG/TextING)) 以及 [《Graph Convolutional Networks for Text Classification》](https://arxiv.org/abs/1809.05679)([TextGCN的代码](https://github.com/yao8839836/text_gcn))。
谢谢他们的工作。

<div align=center>
<img src="https://user-images.githubusercontent.com/59757561/111023162-67c25980-8412-11eb-8c70-0c0b28849fe4.png">
</div>


# 环境配置
python3.7
```bash
conda create -n textseg python=3.7
conda activate textseg
```
python依赖库
```bash
pip install -r requirements.txt
```

# 需要的资源
word2vec:
https://drive.google.com/a/audioburst.com/uc?export=download&confirm=zrin&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

WIKI-727K, WIKI-50:
https://www.dropbox.com/sh/k3jh0fjbyr0gw0a/AADzAd9SDTrBnvs1qLCJY5cza?dl=0

WIKI-SECTION:
https://github.com/sebastianarnold/WikiSection

CHOI:
https://github.com/koomri/text-segmentation/tree/master/data/choi

CLINICAL:
https://github.com/pinkeshbadjatiya/neuralTextSegmentation/tree/master/code/data/clinical

CITIES:
http://groups.csail.mit.edu/rbg/code/mallows/data/wikicities-english.tar.gz

ELEMENTS:
http://groups.csail.mit.edu/rbg/code/mallows/data/wikielements.tar.gz

MANIFESTO:
https://github.com/koomri/text-segmentation/tree/master/data/manifesto

File paths of WIKI-10K we used:
https://drive.google.com/drive/folders/1dYPGOBhXK3kY6ib5pmIKEVmZ5athMMXO

# Usage
Start training as:
```python
python run.py [--train_data_type {wiki,choi,wikisection}]
              [--etype {one_hot,glove,randn,w2v,bert}] [--encoder_fine_tune]
              [--sr_choose {f_model,l_model,s_model,t_model,g_model,b_model,random_baseline}]
              [--tr_choose {balanced,left,right}]
              [--gr_choose {textgcn,texting,ING_GCN,S_ING_GCN}]
              [--gnn_window_size GNN_WINDOW_SIZE]
              [--texting_gru_step TEXTING_GRU_STEP]
```
Example:
```python
python run.py --train_data_type=wiki --etype=w2v --sr_choose=g_model --gr_choose=ING_GCN
```
