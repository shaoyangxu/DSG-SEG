from utils import *
import os
from argparse import ArgumentParser
def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--logger_name', help='logger name', default='seg')
    parser.add_argument('--early_stop', help='early stopping epoch', type = int, default=5)
    parser.add_argument('--train_bs', help='Batch size', type=int, default=8)
    parser.add_argument('--dev_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=100)
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--wiki_random', action='store_false')
    # saved_dataset&saved_encoder
    parser.add_argument("--dataset_dir",type=str, default="/p300/for_git/saved_dataset")
    parser.add_argument("--encoder_dir", type=str, default="/p300/for_git/saved_encoder")
    # frac&filter
    parser.add_argument("--data_frac", type=float, default="1.0")
    parser.add_argument("--length_filter", type=int, default=40000)
    parser.add_argument("--sent_num_filter", type=int, default=60000)
    # data path
    parser.add_argument("--glove_path", type=str, default="/p300/for_git/glove.6B.300d.txt")
    parser.add_argument("--w2v_path", type=str, default="/p300/for_git/GoogleNews-vectors-negative300.bin")
    parser.add_argument("--wiki_path",type=str, default="/p300/for_git/wiki_727")
    parser.add_argument("--choi_path", type=str, default="/p300/for_git/choi")
    parser.add_argument("--manifesto_path", type=str, default="/p300/for_git/manifesto")
    parser.add_argument("--clinical_path", type=str, default="/p300/for_git/clinical")
    parser.add_argument("--wikisection_path", type=str, default="/p300/for_git/wikisection")
    parser.add_argument("--wiki50_path", type=str, default="/p300/for_git/wiki_50")
    parser.add_argument("--wikicities_path", type=str, default="/p300/for_git/wikicities")
    parser.add_argument("--wikielements_path", type=str, default="/p300/for_git/wikielements")
    parser.add_argument("--train_data_type", type=str, default="wiki")
    # Encoder
    parser.add_argument("--etype", type=str,choices=["one-hot","glove","randn","w2v","bert"],default="w2v")
    parser.add_argument("--encoder_fine_tune", action = "store_true") # for bert
    # model
    parser.add_argument("--sr_choose", type=str,choices =["f_model","l_model","s_model","t_model","g_model","b_model","random_baseline"] , default="s_model")
    parser.add_argument('--hidden_dim', type=int, default = 256)
    parser.add_argument('--num_layers', type=int, default = 2)
    # pooling for bilstm & treelstm
    parser.add_argument("--pool_method",type=str,default="max") # max mean attention 'None'
    parser.add_argument('--attention_hd', type=int, default=300)
    # treelstm
    parser.add_argument('--tr_choose', type=str,choices=["balanced","left","right"], default='balanced')
    parser.add_argument("--use_leaf_rnn",action='store_true')
    parser.add_argument("--bidirectional",action='store_true')
    # bilstm
    parser.add_argument('--bilstm_hd', help="hidden dim",type=int, default = 256)
    parser.add_argument('--bilstm_nl', help="num layers",type=int, default = 2)
    parser.add_argument('--bilstm_bd', help="bidirectional", type=bool, default=True)
    # bert
    parser.add_argument('--model_size', help='model size',type=str, default='base')
    parser.add_argument('--cased', help='cased', action='store_false')
    # gnn
    parser.add_argument('--tfidf',action='store_true')
    parser.add_argument('--pmi', action='store_false')
    parser.add_argument('--gr_choose', choices=["texting","DSG_SEG"],type=str, default='texting')
    parser.add_argument('--gnn_window_size', type=int, default=3)
    # texting
    parser.add_argument('--texting_hidden_dim', type=int, default=300)
    parser.add_argument('--texting_output_dim', type=int, default=300)
    parser.add_argument('--texting_gru_step', type=int, default=2)
    # DSG_SEG
    parser.add_argument('--DSG_SEG_output_dim', type=int, default=300)
    # multi-gpu
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    # test
    parser.add_argument('--infer', help='inference_dir', action = "store_true")
    parser.add_argument('--train_checkpoint_dir', help='best model path', default=None)
    parser.add_argument('--best_threshold', help='models trained before were saved in a different way~', default=None)
    args = parser.parse_args()

    args.valid_data_type = args.train_data_type
    checkpoint_name = compute_name(args)
    dataset_name = compute_dataset_name(args)
    encoder_name = compute_encoder_name(args)
    if args.train_checkpoint_dir != None:
        args.train_checkpoint_dir = os.path.join(args.checkpoint_dir, args.train_checkpoint_dir)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, checkpoint_name)
    # dataset
    args.dataset_path = os.path.join(args.dataset_dir, dataset_name + ".pt")
    # encoder
    if args.encoder_fine_tune:
        args.encoder_path = os.path.join(args.encoder_dir, encoder_name + ".pt")
    else:
        args.encoder_path = os.path.join(args.encoder_dir, encoder_name + ".pt")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.encoder_dir, exist_ok=True)
    return args
