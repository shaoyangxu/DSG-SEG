import torch.nn as nn
import torch
from utils import *
from collections import OrderedDict
import numpy as np
class glove_model(nn.Module):

    def __init__(self, vocab_path, args, encoder_fine_tune):
        super(glove_model,self).__init__()
        self.vocab_path = vocab_path
        self.args = args
        self.encoder_fine_tune = encoder_fine_tune
        self.local_rank = args.local_rank
        self.PADDING_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.embedding, self.word2idx = self.get_glove_embedding()
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

    def get_glove_embedding(self):
        size = 300
        logger = get_logger(self.args)
        rank_logger_info(logger, self.local_rank, 'Reading glove Vocab.')
        glove_word2idx = OrderedDict()
        glove_word2idx[self.PADDING_TOKEN] = len(glove_word2idx)
        glove_word2idx[self.UNK_TOKEN] = len(glove_word2idx)

        with open(self.vocab_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                if word in glove_word2idx:
                    continue
                else:
                    glove_word2idx[word] = len(glove_word2idx)

        embeddings = np.zeros((len(glove_word2idx), size), dtype=np.float32)
        embeddings[0] = np.random.randn(1, size)
        embeddings[1] = np.random.randn(1, size)
        rank_logger_info(logger, self.local_rank, 'Reading glove Embeddings.')
        with open(self.vocab_path, encoding='utf-8') as f:
            for i,line in enumerate(f):
                word, vec = line.strip().split(' ', 1)
                if word not in glove_word2idx:
                    continue
                word_id = glove_word2idx[word]
                vec = np.fromstring(vec, dtype=float, sep=' ')
                embeddings[word_id] = vec

        return torch.tensor(embeddings), glove_word2idx

    def tokenize(self, inp):
        word_lst = inp.split()
        idx_lst = []
        for word in word_lst:
            if word in self.word2idx:
                idx_lst.append(self.word2idx[word])
            else:
                idx_lst.append(self.word2idx[self.UNK_TOKEN])
        return idx_lst


    def forward(self,inp):
        return self.embedding(inp)

    def get_word_dim(self):
        return 300