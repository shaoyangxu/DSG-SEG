import torch.nn as nn
import torch
from utils import *
from collections import OrderedDict
import numpy as np
class one_hot_model(nn.Module):

    def __init__(self, vocab_path, args, encoder_fine_tune):
        super(one_hot_model,self).__init__()
        self.vocab_path = vocab_path
        self.args = args
        self.encoder_fine_tune = encoder_fine_tune
        self.local_rank = args.local_rank
        self.PADDING_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.embedding, self.word2idx = self.get_one_hot_embedding()
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

    def get_one_hot_embedding(self):
        logger = get_logger(self.args)
        rank_logger_info(logger, self.local_rank, 'Reading one-hot Vocab.')

        one_hot_word2idx = OrderedDict()
        one_hot_word2idx[self.PADDING_TOKEN] = len(one_hot_word2idx)
        one_hot_word2idx[self.UNK_TOKEN] = len(one_hot_word2idx)

        with open(self.vocab_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 10000:
                    break
                word, vec = line.split(' ', 1)
                if word in one_hot_word2idx:
                    continue
                else:
                    one_hot_word2idx[word] = len(one_hot_word2idx)
        rank_logger_info(logger, self.local_rank, 'Reading one-hot embeddings.')
        embeddings = np.zeros((len(one_hot_word2idx) , len(one_hot_word2idx)), dtype=np.float32)
        for i in range(len(one_hot_word2idx)):
            embeddings[i][i] = 1
        return torch.tensor(embeddings), one_hot_word2idx

    def forward(self, inp):
        return self.embedding(inp)

    def tokenize(self, inp):
        word_lst = inp.split()
        idx_lst = []
        for word in word_lst:
            if word in self.word2idx:
                idx_lst.append(self.word2idx[word])
            else:
                idx_lst.append(self.word2idx[self.UNK_TOKEN])
        return idx_lst

    def get_word_dim(self):
        return len(self.word2idx)


