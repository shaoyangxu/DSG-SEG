from tqdm import tqdm
import torch
import json
from evaluation_utils import *
from utils import *
import torch.nn.functional as F
from termcolor import colored
import random
preds_stats = predictions_analysis()
def test(model,encode_model, args, dataset,threshold,local_rank):
    logger = get_logger(args)
    device = args.device
    global output_seg, current_idx
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        acc = Accuracy() # accuracies class is not needed
        document_result = {}
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                with torch.no_grad():
                    encoded_input = encode_model(data)
                    output = model(encoded_input)
                    output_seg = output.data.cpu().numpy().argmax(axis=1)
                    targets_var = torch.cat(target, 0).to(device).clone().detach().requires_grad_(False)
                    target_seg = targets_var.data.cpu().numpy()
                    preds_stats.add(output_seg, target_seg)
                    output_softmax = F.softmax(output, 1)
                    current_idx = 0
                    for k, t in enumerate(target):
                        path = paths[k]
                        document_sentence_count = len(t)
                        to_idx = int(current_idx + document_sentence_count)
                        output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold).tolist()
                        if "_|_" not in path:  # whole document
                            document_result[path] = [[output,t]] # [[[1,0,1,0],[1,0,1,0]],....]
                        else: # filtered document
                            path,num = path.split("_|_")
                            if path not in document_result:
                                document_result[path] = [[num,[output,t]]]
                            else:
                                document_result[path].append([num,[output,t]])
                        current_idx = to_idx
        # concat
        for k in document_result:
            if type(document_result[k][0][0]) == str:
                splited_dic_item = document_result[k]
                complete_dic_item = []
                sorted_splited_dic_item = sorted(splited_dic_item, key=lambda x: int(x[0]))
                for item in sorted_splited_dic_item:
                    if len(complete_dic_item) == 0:
                        complete_dic_item = [item[1][0], [item[1][1]]]
                    else:
                        complete_dic_item[0].extend(item[1][0])
                        complete_dic_item[1].append(item[1][1])

                document_result[k] = complete_dic_item
                document_result[k][1] = torch.cat(document_result[k][1], 0)
            else:
                document_result[k] = [document_result[k][0][0], document_result[k][0][1]] #[[output],[target]]
            document_result[k][0] = np.append(document_result[k][0],[1])
            document_result[k][1] = np.append(document_result[k][1],[1])
        for k in document_result:
            acc.update(document_result[k][0], document_result[k][1], k)
        epoch_pk, epoch_windiff, epoch_b, epoch_s = acc.calc_accuracy()
        rank_logger_info(logger, local_rank, colored('accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, B: {:.4}, S: {:.4} ,F1: {:.4} . '.format(
                                                                                                     preds_stats.get_accuracy(),
                                                                                                    epoch_pk,
                                                                                                    epoch_windiff,
                                                                                                    epoch_b,
                                                                                                    epoch_s,
                                                                                                    preds_stats.get_f1()),"red"))
        preds_stats.reset()
        epoch_result = acc.all_test_result
        return epoch_pk, epoch_result

def test_all(model, encode_model, args, test_dl_dict, threshold, local_rank):
    logger = get_logger(args)
    pk_result = {}
    for test_data_type in test_dl_dict:
        test_dl = test_dl_dict[test_data_type]
        rank_logger_info(logger, local_rank, f"Testing on {test_data_type}...")
        test_pk, test_result = test(model,encode_model, args, test_dl, threshold, local_rank)
        pk_result[test_data_type] = test_pk
        if local_rank in [-1, 0]:
            with open(args.checkpoint_dir / f"{test_data_type}_best_result.json", "w") as f:
                json.dump(test_result, f)
    flags = {k: v for k, v in pk_result.items()}
    rank_logger_info(logger, local_rank, json.dumps(flags, indent=4, sort_keys=True))


def test_all_random_baseline(args, test_dl_dict, local_rank):
    logger = get_logger(args)
    pk_result = {}
    avg_sent_num_per_seg = {}  # all_sent_num / all_seg_num
    avg_sent_num_per_doc = {}  # all_sent_num / all_doc_num
    threshold_dict = {}
    avg_seg_num_per_doc = {}  # all_seg_num / all_doc_num
    avg_sentlength_per_sent = {} # all_sent_length / all_sent_num
    for test_data_type in test_dl_dict:
        all_doc_num = 0
        all_sent_num = 0
        all_seg_num = 0
        all_sent_length = 0
        test_dl = test_dl_dict[test_data_type]
        rank_logger_info(logger, local_rank, f"Getting threshold on {test_data_type}...")
        for i, (data, target, paths) in enumerate(test_dl): # batch size = 1
            all_doc_num += len(target)
            for idx,t in enumerate(target):
                all_sent_num += len(t)
                all_seg_num += torch.sum(t).item()
                for sent in data[idx]:
                    all_sent_length += len(sent)
        avg_sent_num_per_seg[test_data_type] = all_sent_num / all_seg_num
        threshold_dict[test_data_type] = 1-1/avg_sent_num_per_seg[test_data_type]
        avg_seg_num_per_doc[test_data_type] = all_seg_num / all_doc_num
        avg_sent_num_per_doc[test_data_type] = all_sent_num / all_doc_num
        avg_sentlength_per_sent[test_data_type] = all_sent_length / all_sent_num
    rank_logger_info(logger, local_rank, f"avg_sent_num_per_seg:")
    rank_logger_info(logger, local_rank, json.dumps(avg_sent_num_per_seg, indent=4, sort_keys=True))
    rank_logger_info(logger, local_rank, f"avg_seg_num_per_doc:")
    rank_logger_info(logger, local_rank, json.dumps(avg_sent_num_per_doc, indent=4, sort_keys=True))
    rank_logger_info(logger, local_rank, f"avg_sent_num_per_doc:")
    rank_logger_info(logger, local_rank, json.dumps(avg_seg_num_per_doc, indent=4, sort_keys=True))
    rank_logger_info(logger, local_rank, f"avg_sentlength_per_sent:")
    rank_logger_info(logger, local_rank, json.dumps(avg_sentlength_per_sent, indent=4, sort_keys=True))
    rank_logger_info(logger, local_rank, f"threshold:")
    rank_logger_info(logger, local_rank, json.dumps(threshold_dict, indent=4, sort_keys=True))
    for test_data_type in test_dl_dict:
        rank_logger_info(logger, local_rank, f"Getting random baseline result on {test_data_type}...")
        threshold = threshold_dict[test_data_type]
        acc = Accuracy()
        test_dl = test_dl_dict[test_data_type]
        for i, (data, target, paths) in enumerate(test_dl):  # batch size = 1
            for idx,t in enumerate(target):
                h = []
                for j in range(len(t)):
                    random_seed = random.random()
                    if random_seed > threshold:
                        h.append(1)
                    else:
                        h.append(0)
                gold = np.append(t, [1])
                h = np.append(h,[1])
                acc.update(h, gold, paths[idx])
        epoch_pk, epoch_windiff, epoch_b, epoch_s = acc.calc_accuracy()
        pk_result[test_data_type] = epoch_pk
    rank_logger_info(logger, local_rank, json.dumps(pk_result, indent=4, sort_keys=True))



