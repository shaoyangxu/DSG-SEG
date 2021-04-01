# DSG-SEG整体代码结构  

DSG-SEG 的整体目录结构介绍如下：  
```
DSE-SEG
├── doc                                             //介绍文档
├── data                                            //加载数据集
│   ├── all_data_loader.py                          //汇总
│   ├── choi_loader.py                              //choi
│   ├── clinical_loader.py                          //clinical
│   ├── data_utils.py                               //工具
│   ├── manifesto_loader.py                         //manifesto
│   ├── wiki_loader.py                              //wiki
│   ├── wiki50_loader.py                            //wiki50
│   ├── wikicities_loader.py                        //wikicities
│   ├── wikielements_loader.py                      //wikielements
│   └── wikisection_loader.py                       //wikisection
├── models                                          //模型
│   ├── GGNN                                        //GGNN网络
│   │   ├── init.py                                 //四种参数初始化方式
│   │   └── layer.py                                //网络
│   ├── Treelstm                                    //Treelstm模型
│   │   ├── basic.py                                //功能
│   │   ├── data.py                                 //数据
│   │   ├── models.py                               //模型
│   │   └── utils.py                                //工具
│   ├── bert_sr.py                                  //Bert句子表示
│   ├── DSG_SEG.py                                  //DSG_SEG句子表示
│   ├── Encode_model.py                             //编码模型，数据流向:dataset->collate_fn->data_loader->encoder_model->sr_model->seg_model
│   ├── freq_sr.py                                  //词频句子表示
│   ├── latent_sr.py                                //隐式句子表示
│   ├── seg_model.py                                //上层Bilstm网络，用于预测分割点
│   ├── sequence_sr.py                              //序列句子表示
│   ├── texting                                     //texting模型，也就是-SENTNODE消融模型
│   └── treelstm_sr                                 //树形句子表示
├── encoder                                         //词向量
│   ├── w2v_encoder.py                              //word2vec
│   ├── randn_encoder.py                            //随机初始化
│   ├── one_hot_encoder.py                          //one-hot
│   ├── glove_encoder.py                            //glove
│   ├── Encoder.py                                  //汇总
│   └── bert_encoder.py                             //bert
├── validate.py                                     //模型验证
├── utils.py                                        //工具
├── train.py                                        //模型训练
├── test.py                                         //模型测试
├── run.py                                          //主程序
├── parameters.py                                   //命令行参数
├── judgement.py                                    //样例分析
└── evaluation_utils.py                             //评估工具
```
