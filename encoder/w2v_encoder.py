import torch.nn as nn
import torch
from utils import *
from collections import OrderedDict
import numpy as np
import gensim
class w2v_model(nn.Module):

    def __init__(self, vocab_path, args, encoder_fine_tune):
        super(w2v_model,self).__init__()
        self.vocab_path = vocab_path
        self.args = args
        self.encoder_fine_tune = encoder_fine_tune
        self.local_rank = args.local_rank
        self.PADDING_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.embedding, self.word2idx = self.get_w2v_embedding()
        self.pad_token_id = self.word2idx[self.PADDING_TOKEN]
        self.idx2word = OrderedDict()
        for item in self.word2idx.items():
            self.idx2word[item[1]] = item[0]
        self.embedding = nn.Embedding.from_pretrained(self.embedding)
        if encoder_fine_tune:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = False

    def get_w2v_embedding(self):
        size = 300
        logger = get_logger(self.args)
        rank_logger_info(logger, self.local_rank, 'Reading w2v Vocab.')
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.vocab_path, binary=True)

        w2v_word2idx = OrderedDict()
        w2v_word2idx[self.PADDING_TOKEN] = len(w2v_word2idx)
        # w2v_word2idx[self.UNK_TOKEN] = len(w2v_word2idx)

        for w in word2vec.vocab:
            if w in w2v_word2idx:
                continue
            else:
                w2v_word2idx[w] = len(w2v_word2idx)

        embeddings = np.zeros((len(w2v_word2idx), size), dtype=np.float32)
        embeddings[0] = np.zeros((1, size))
        # embeddings[1] = np.random.randn(1, size)
        rank_logger_info(logger, self.local_rank, 'Reading w2v Embeddings.')
        for word in word2vec.vocab:
            vec = word2vec[word]
            if word not in w2v_word2idx:
                continue
            word_id = w2v_word2idx[word]
            embeddings[word_id] = vec

        return torch.tensor(embeddings), w2v_word2idx

    def tokenize(self, inp):
        word_lst = inp.split()
        idx_lst = []
        for word in word_lst:
            if word in self.word2idx:
                idx_lst.append(self.word2idx[word])
            else:
                idx_lst.append(self.word2idx['UNK']) # self.word2idx[self.UNK_TOKEN]
        return idx_lst

    def forward(self,inp):
        return self.embedding(inp)
    def get_word_dim(self):
        return 300
