import torch
from torch.utils.data import DataLoader
import os
from pathlib2 import Path
from termcolor import colored
import re
from train import train
from test import test_all, test_all_random_baseline 
from validate import validate
from encoder.Encoder import Encoder
from models.Encode_model import Encode_model
from data.all_data_loader import get_loader
from utils import *
from models.seg_model import seg_model
from parameters import *
from torch.utils.data.distributed import DistributedSampler
test_data_type_list = ['wiki','choi','clinical', 'wikielements','wikicities','wiki50', 'manifesto'] # wikisection

def main(args):
    global test_data_type_list
    logger = get_logger(args)
    args.checkpoint_dir = Path(args.checkpoint_dir)
    # initial encoder
    encoder = None
    if not os.path.exists(args.encoder_path):
        rank_logger_info(logger, args.local_rank, "Creating Encoder...")
        encoder = Encoder(args=args, etype=args.etype, encoder_fine_tune=args.encoder_fine_tune, glove_path=args.glove_path, w2v_path=args.w2v_path)
        if args.local_rank in [-1, 0]:
            torch.save(encoder, args.encoder_path)
    else:
        rank_logger_info(logger, args.local_rank, "Loading Encoder...")
        encoder = torch.load(args.encoder_path)
    encode_model = Encode_model(encoder, args)
    # initial dataset
    train_dataset, dev_dataset,test_dataset_dict  = None,None,{}
    if not os.path.exists(args.dataset_path): # create
        if not args.infer: # train:read all dataset
            rank_logger_info(logger, args.local_rank, "Creating dataset...")
            train_dataset = get_loader(args=args, encoder = encoder, data_type=args.train_data_type, split="train")
            dev_dataset = get_loader(args=args, encoder = encoder, data_type=args.valid_data_type, split="dev")
            for test_data_type in test_data_type_list:
                if test_data_type in ["wikisection", "wiki"]:
                    test_dataset_dict[test_data_type] = get_loader(args=args, encoder = encoder, data_type=test_data_type, split="test")
                else:
                    test_dataset_dict[test_data_type] = get_loader(args=args, encoder=encoder, data_type=test_data_type)
            saved_pt = {
                "train_dataset":train_dataset,
                "dev_dataset":dev_dataset,
                "test_dataset":None # test_dataset_dict
            }
            if args.local_rank in [-1, 0]:
                torch.save(saved_pt, args.dataset_path)
        else: # infer:just read test dataset and not save
            for test_data_type in test_data_type_list:
                if test_data_type in ["wikisection", "wiki"]:
                    test_dataset_dict[test_data_type] = get_loader(args=args, encoder = encoder, data_type=test_data_type, split="test")
                else:
                    test_dataset_dict[test_data_type] = get_loader(args=args, encoder=encoder, data_type=test_data_type)
    else: # load
        rank_logger_info(logger, args.local_rank, "Loading dataset...")
        loaded_pt = torch.load(args.dataset_path)
        train_dataset, dev_dataset,test_dataset_dict = loaded_pt["train_dataset"], loaded_pt["dev_dataset"],loaded_pt["test_dataset"]
    # initial dataloader
    rank_logger_info(logger, args.local_rank, "Creating dataloader...")
    collate_fn = get_collate_fn(args)
    if not args.infer:
        train_sampler = DistributedSampler(train_dataset) if args.local_rank != - 1 else None
        train_dl = DataLoader(train_dataset, batch_size=args.train_bs, collate_fn=collate_fn, shuffle = True if not train_sampler else False,
                              sampler=train_sampler, num_workers=args.num_workers)
        dev_dl = DataLoader(dev_dataset, batch_size=args.dev_bs, collate_fn=collate_fn, shuffle=False,num_workers=args.num_workers)
    test_dl_dict = {}
    for test_data_type in test_data_type_list:
        test_dl_dict[test_data_type] = DataLoader(test_dataset_dict[test_data_type], batch_size=args.test_bs, collate_fn=collate_fn, shuffle = False,
                        num_workers=args.num_workers)
    rank_logger_info(logger, args.local_rank, "Creating model...")
    model = seg_model(sr_choose=args.sr_choose, encoder_input_dim=encode_model.encoder.get_input_dim(), args=args)
    model.to(args.device)
    encode_model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
        encode_model = torch.nn.parallel.DistributedDataParallel(encode_model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

    rank_logger_info(logger, args.local_rank, "Creating optimizer...")
    params = []
    trainable_params_num = 0
    for name, param in list(encode_model.named_parameters()) + list(model.named_parameters()):
        if param.requires_grad:
            params.append(param)
            trainable_params_num += param.numel()
            rank_logger_info(logger, args.local_rank,f"name: {name} | param_num: {param.numel()}")
    rank_logger_info(logger, args.local_rank, f"trainable_params_num: {trainable_params_num}")
    optimizer = getattr(torch.optim, "Adam")(params, lr=1e-3) #
    rank_logger_info(logger, args.local_rank, model)
    rank_logger_info(logger, args.local_rank, "Device on:" + str(args.device))
    # train
    if not args.infer:
        model.train()
        best_val_pk,early_stop_cot,best_epoch,best_threshold = 1.0,0,0,0
        for j in range(args.epochs):
            train(model=model,encode_model=encode_model, args=args, epoch=j, dataset=train_dl, optimizer=optimizer,local_rank= args.local_rank)
            val_pk, threshold = validate(model,encode_model,args, j, dev_dl,local_rank=args.local_rank)
            if val_pk < best_val_pk:
                best_val_pk,early_stop_cot,best_epoch,best_threshold = val_pk,0,j+1,threshold
                rank_logger_info(logger, args.local_rank, colored(
                        'Current best model from epoch {} with p_k {} and threshold {}'.format(j+1, best_val_pk, threshold),'red'))
                rank_logger_info(logger, args.local_rank,f"saveing best model with p_k {best_val_pk} and threshold {best_threshold}")
                saved_dict = {
                    "best_model":model,
                    "best_val_pk":best_val_pk,
                    "best_epoch":best_epoch,
                    "best_threshold":best_threshold
                }
                if args.local_rank in [-1, 0]:
                    with (args.checkpoint_dir / 'best_model.t7').open('wb') as f:
                        torch.save(saved_dict, f)
                    if args.etype == "bert": # only bert encoder may be finetuned
                        with (args.checkpoint_dir / 'best_encode_model.t7').open('wb') as f:
                            torch.save(encode_model, f)
            else:
                early_stop_cot += 1
                rank_logger_info(logger, args.local_rank,"pk of epoch {} didnt decrease, early_stop_cot is {} now".format(j+1,early_stop_cot))
                if early_stop_cot == args.early_stop:
                    rank_logger_info(logger, args.local_rank, 'early stopping for epoch {}'.format(j+1))
                    break
        # test
        rank_logger_info(logger, args.local_rank, "Loading best model(and encode model)...")
        best_dict = torch.load(str(args.checkpoint_dir / 'best_model.t7'))
        best_model, best_threshold, best_encode_model = best_dict["best_model"],best_dict["best_threshold"],torch.load(str(args.checkpoint_dir / 'best_encode_model.t7')) if args.etype == "bert" else encode_model
        rank_logger_info(logger, args.local_rank, "Testing...")
        test_all(best_model, best_encode_model, args, test_dl_dict,threshold = best_threshold, local_rank=args.local_rank)
    else:
        if args.sr_choose == 'random_baseline':
            rank_logger_info(logger, args.local_rank, "Getting random baseline results...")
            test_all_random_baseline(args, test_dl_dict, local_rank=args.local_rank)
            exit(1)
        train_checkpoint_dir = Path(re.sub("_infer","",str(args.checkpoint_dir))) if args.train_checkpoint_dir == None else Path(args.train_checkpoint_dir)
        rank_logger_info(logger, args.local_rank, f"Loading best model(and encode model) from {str(train_checkpoint_dir)}...")
        if os.path.exists(str(train_checkpoint_dir / 'best_model.t7')):
            try:
                best_dict = torch.load(str(train_checkpoint_dir / 'best_model.t7'))
                best_model, \
                best_threshold, \
                best_encode_model = best_dict["best_model"], \
                                    best_dict["best_threshold"], \
                                    torch.load(str(train_checkpoint_dir / 'best_encode_model.t7')) if args.etype == "bert" else encode_model
            except:
                best_model = torch.load(str(train_checkpoint_dir / 'best_model.t7'))
                assert args.best_threshold != None
                best_threshold = float(args.best_threshold)
                best_encode_model = torch.load(str(train_checkpoint_dir / 'best_encode_model.t7')) if args.etype == "bert" else encode_model
            rank_logger_info(logger, args.local_rank, "Testing...")
            test_all(best_model, best_encode_model, args, test_dl_dict, threshold=best_threshold,local_rank=args.local_rank)
        else:
            rank_logger_info(logger, args.local_rank, "No existing best model(and encode model)!!!")
            exit(1)

if __name__ == '__main__':
    args = create_parser()
    set_seed(args.seed)
    logger = setup_logger(args.logger_name, os.path.join(args.checkpoint_dir, 'train.log'))
    rank_logger_info(logger, args.local_rank, stringify_flags(args))
    args.device = set_multi_gpu(args)
    main(args)