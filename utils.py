import json
import logging
import sys
import torch
import math
from collections import OrderedDict
import hashlib
import random
import numpy as np
import torch.distributed as dist

PADDING_TOKEN = "PAD"
UNK_TOKEN = "UNK"


def set_multi_gpu(args):
    if args.local_rank != -1:
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend)
    device = args.local_rank if args.local_rank != -1 else (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    torch.cuda.set_device(args.local_rank)
    return device

def rank_logger_info(logger, rank, info):
    if rank in [-1, 0]:
        logger.info(info)
    else:
        pass

def set_seed(seed):
    random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_logger(args):
    return logging.getLogger(args.logger_name)

def compute_name(args):
    imp_opts = ["seed",
                "train_data_type",
                "train_bs","dev_bs","test_bs",
                "data_frac","length_filter","sent_num_filter",
                "etype","encoder_fine_tune","model_size","cased",
                "sr_choose","tr_choose","gr_choose",
                "hidden_dim","num_layers","pool_method","attention_hd","use_leaf_rnn","bidirectional",
                "bilstm_hd","bilstm_nl","bilstm_bd",
                "model_size"]
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    hp_dict = vars(args)
    for key in imp_opts:
        val = hp_dict[key]
        opt_dict[key] = val
    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    pre_name = str(opt_dict["seed"]) + "_" + opt_dict["etype"]
    if args.sr_choose == 'g_model':
        pre_name += "_"
        pre_name += opt_dict["gr_choose"]
    elif args.sr_choose == 't_model':
        pre_name += "_"
        pre_name += opt_dict["tr_choose"]
    else:
        pre_name += "_"
        pre_name += opt_dict["sr_choose"]
    if not args.infer:
        return pre_name + "_" + str(hash_idx)
    else:
        return pre_name + "_" + "infer" + "_" + str(hash_idx)

def compute_dataset_name(args):
    imp_opts=["train_data_type","data_frac","length_filter","sent_num_filter","etype"]
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    hp_dict = vars(args)
    for key in imp_opts:
        val = hp_dict[key]
        opt_dict[key] = val
    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    if args.sr_choose == "g_model":
        if args.gr_choose == 'DSG_SEG':
            name = "gcn" + "_" + str(hash_idx)
        elif args.gr_choose == 'texting':
            name = "ing" + "_" + str(hash_idx)
    else:
        name = str(hash_idx)
    return args.train_data_type + "_" + name

def compute_encoder_name(args):
    name = args.etype
    if args.encoder_fine_tune:
        name += "_ft"
    return name

def stringify_flags(options):
    # Ignore negative boolean flags.
    flags = {k: v for k, v in options.__dict__.items()}
    return json.dumps(flags, indent=4, sort_keys=True)

def setup_logger(logger_name, filename, delete_old = False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler   = logging.FileHandler(filename, mode='w') if delete_old else logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger

def unsort(sort_order):
    result = [-1] * len(sort_order)

    for i, index in enumerate(sort_order):
        result[index] = i

    return result

def get_collate_fn(args):
    if args.sr_choose == 'g_model' and args.gr_choose == 'texting':
        return ing_collate_fn
    elif args.sr_choose == 'g_model' and args.gr_choose == 'DSG_SEG':
        return gcn_collate_fn
    else:
        return collate_fn

def collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []
    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) / 2))
    after_sentence_count = window_size - before_sentence_count - 1
    for example in batch:
        data, targets, path = example["sentences"], example["targets"], example["path"]
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in range(0, len(data)):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                tensored_data.append(torch.tensor(sentences_window))
            tensored_targets = torch.tensor(targets).long()
            # tensored_targets = tensored_targets[:-1] # removed in wiki_loader.py
            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue
    return batched_data, batched_targets, paths

def ing_collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []
    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) / 2))
    after_sentence_count = window_size - before_sentence_count - 1
    # batch: [[[tensor(sent_num1,), tensor(sent_num1,sent_num1)], ...], [doc2], [doc3], []]
    for example in batch:
        data, adjs, targets, path = example["sentences"],example["adjs"],example["targets"], example["path"]
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in range(0, len(data)):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                adj_window = adjs[from_index:to_index][0] # fault window size = 1
                tensored_data.append([torch.tensor(sentences_window), torch.tensor(adj_window).float()])
            tensored_targets = torch.tensor(targets).long()
            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue
    return batched_data, batched_targets, paths

def gcn_collate_fn(batch):
    batched_data = [] # [[sentences, feature, adj], [], [], [],]
    batched_targets = []
    paths = []
    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) / 2))
    after_sentence_count = window_size - before_sentence_count - 1
    for example in batch:
        sentences, feature, adjs, targets, path = example["sentences"],example["feature"],example["adj"],example["targets"],example["path"]
        try:
            max_index = len(sentences)
            sentences_lst = []
            for curr_sentence_index in range(0, len(sentences)):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in sentences[from_index:to_index] for word in sentence]
                sentences_lst.append(sentences_window)
            doc_lst = [sentences_lst, torch.tensor(feature), torch.tensor(adjs.A).float()]
            batched_data.append(doc_lst)
            tensored_targets = torch.tensor(targets).long()
            batched_targets.append(tensored_targets)
            paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue
    return batched_data, batched_targets, paths