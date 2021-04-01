import torch.nn as nn
import torch
from utils import *
from collections import OrderedDict
import numpy as np

class randn_model(nn.Module):

    def __init__(self, vocab_path, args,encoder_fine_tune):
        super(randn_model, self).__init__()
        self.vocab_path = vocab_path
        self.args = args
        self.encoder_fine_tune = encoder_fine_tune
        self.local_rank = args.local_rank
        self.PADDING_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.embedding, self.word2idx = self.get_randn_embedding()
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

    def get_randn_embedding(self):
        logger = get_logger(self.args)
        rank_logger_info(logger, self.local_rank, 'Reading randn Vocab.')

        randn_word2idx = OrderedDict()
        randn_word2idx[self.PADDING_TOKEN] = len(randn_word2idx)
        randn_word2idx[self.UNK_TOKEN] = len(randn_word2idx)

        with open(self.vocab_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                if word in randn_word2idx:
                    continue
                else:
                    randn_word2idx[word] = len(randn_word2idx)
        rank_logger_info(logger, self.local_rank, 'Reading randn embedding.')
        embeddings = np.random.randn(len(randn_word2idx), 300)
        return torch.tensor(embeddings).float(), randn_word2idx

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
        return 300
