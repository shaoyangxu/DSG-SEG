# 《基于篇章结构图网络的话题分割》(Topic Segmentation via Discourse Structure Graph Network)
该仓库包含了《基于篇章结构图网络的话题分割》文章的代码和数据集，代码使用pytorch框架实现。
我们的工作主要基于[《Text Segmentation as a Supervised Learning Task》](https://arxiv.org/abs/1803.09337) ，它的源代码是[text-segmentation](https://github.com/koomri/text-segmentation)。
另外，我们的模型灵感主要来自于 [《Every Document Owns Its Structure: Inductive Text Classification via
Graph Neural Networks》](https://arxiv.org/abs/2004.13826)([TextING的代码](https://github.com/CRIPAC-DIG/TextING)) 以及 [《Graph Convolutional Networks for Text Classification》](https://arxiv.org/abs/1809.05679)([TextGCN的代码](https://github.com/yao8839836/text_gcn))。
谢谢他们的工作。

# 论文简介

<div align=center>
<img src="https://user-images.githubusercontent.com/59757561/111023162-67c25980-8412-11eb-8c70-0c0b28849fe4.png">
</div>

我们提出了一个基于篇章结构图网络的话题分割模型——DSG-SEG，如上图所示.具体的，模型把每一个篇章单独构建成图，图中包含了单词、句子节点，以及单词节点之间、单词和句子节点之间的邻接关系；接着，模型把图作为输入，对GGNN网络进行迭代，句子节点之间因此通过共有的邻接单词节点产生了间接的信息交互；最终，模型得到了具有全局语义信息的句子向量表示，并将其送入Bi-LSTM网络进行分割点的预测.

我们对比了六种基准模型:latent-sr\latent-sr\Bert+Bi-LSTM\tree-sr-left\TextSeg\Bert+Bi-LSTMnft，我们的模型在诸多数据集上同时取得了最好的结果指标和时间性能.

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

下载并解压WIKI-727K之后，把上面链接中的`dev_random_paths`,`test_random_paths`,`train_random_paths`分别放在`wiki_727K\dev`,`wiki_727K\test`,`wiki_727K\train`路径下，并都改名为`random_paths`

下载完并解压所有数据集，一共得到八个文件夹:wiki_727\choi\wikicites\wiki_50\clinical\manifesto\WikiSection\wikielements，记得在parameters.py中修改对应的路径.

# 代码简介
|文件(夹)名|作用|
|-|-|
|models|所有基准模型(treelstm_sr\latent_sr\freq_sr\sequence_sr=TextSeg\bert_sr=Bert+Bilstm)\DSG_SEG\Encode_model|
|data|8个数据集的加载程序(wiki\choi\clinical\wikisection\wikielements\wikicites\wiki50\manifesto)|
|encoder|5种词向量初始化方式(word2vec\bert\randn\one-hot\glove)|
|run.py|主程序|
|train\validate\test.py|分别定义了训练\验证\测试函数|
|parameters.py|接收外部参数|
|utils.py|工具函数|
|evaluation_utils.py|计算Pk和Accuracy|
|judgement.py|样例分析|


# 代码运行(仅展示部分)
```python
python run.py [--etype {one_hot,glove,randn,w2v,bert}] [--encoder_fine_tune]
              [--sr_choose {f_model,l_model,s_model,t_model,g_model,b_model,random_baseline}]
              [--tr_choose {balanced,left,right}]
              [--gr_choose {texting,DSG_GCN}]
              ...
```
DSG_GCN:
```python
python run.py --sr_choose=g_model --gr_choose=DSG_GCN
```
-SENTNODE:
```python
python run.py --sr_choose=g_model --gr_choose=texting
```
-PMI:
```python
python run.py --sr_choose=g_model --gr_choose=DSG_GCN --pmi
```
Bert+Bi-LSTMnft:
```python
python run.py --etype=bert --sr_choose=b_model --length_filter=64
```
TextSeg:
```python
python run.py --sr_choose=s_model
```
Bert+Bi-LSTM:
```python
python run.py --etype=bert --sr_choose=b_model --length_filter=40 --sent_num_filter=60 --train_bs=3 --encoder_fine_tune
```
tree-sr-left
```python
sr_choose=t_model --tr_choose=left --length_filter=40 --sent_num_filter=60
```
freq-sr:
```python
python run.py --etype=one-hot --sr_choose=f_model
```
latent-sr:
```python
python run.py --etype=one-hot --sr_choose=l_model
```
random-baseline:
```python

```
