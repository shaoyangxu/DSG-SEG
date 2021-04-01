from tqdm import tqdm
import torch
import torch.nn.functional as F
from evaluation_utils import *
from termcolor import colored
from utils import *
import numpy as np
preds_stats = predictions_analysis()
def validate(model, encode_model, args, epoch, dataset,local_rank):
    device = args.device
    logger = get_logger(args)
    model.eval()
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        acc = Accuracies()
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
                    output_softmax = F.softmax(output, 1)
                    current_idx = 0
                    output_np = output_softmax.data.cpu().numpy()
                    for k, t in enumerate(target):
                        path = paths[k]
                        document_sentence_count = len(t)
                        to_idx = int(current_idx + document_sentence_count)
                        tmp_output = output_np[current_idx: to_idx, :].tolist()
                        if "_|_" not in path:  # whole document
                            document_result[path] = [[tmp_output,t]]
                        else: # filtered document
                            path,num = path.split("_|_")
                            if path not in document_result:
                                document_result[path] = [[num,[tmp_output,t]]]
                            else:
                                document_result[path].append([num,[tmp_output,t]])
                        current_idx = to_idx
                targets_var = torch.cat(target, 0).to(device).clone().detach().requires_grad_(False)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)
        # concat:
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
                # complete_dic_item[1] =
                document_result[k] = complete_dic_item
                document_result[k][1] = torch.cat(document_result[k][1], 0)
            else:
                document_result[k] = [document_result[k][0][0],document_result[k][0][1]]
        acc_input = [[],[],[]]  # [all_output_softmax, all_target, all_paths, args.crf]
        for k in document_result:
            acc_input[0].extend(document_result[k][0])
            acc_input[1].append(document_result[k][1])
            acc_input[2].append(k)
        acc_input[0] = np.array(acc_input[0])
        acc.update(acc_input[0], acc_input[1], acc_input[2])
        epoch_pk, epoch_windiff, epoch_b, epoch_s, threshold = acc.calc_accuracy()

        rank_logger_info(logger, local_rank,  colored('Validating Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, B: {:.4}, S: {:.4} ,F1: {:.4} . '.format(epoch + 1,
                                                                                                            preds_stats.get_accuracy(),
                                                                                                            epoch_pk,
                                                                                                            epoch_windiff,
                                                                                                            epoch_b,
                                                                                                            epoch_s,
                                                                                                            preds_stats.get_f1()),"red"))
        preds_stats.reset()

        return epoch_pk, threshold

