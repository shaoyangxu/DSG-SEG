from tqdm import tqdm
import torch
from evaluation_utils import *
from utils import *
from termcolor import colored
import numpy as np
preds_stats = predictions_analysis()
def train(model, encode_model, args, epoch, dataset, optimizer,local_rank=-1):
    device = args.device
    logger = get_logger(args)
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        acc = Accuracy() # accuracies class is not needed
        document_result = {}
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                optimizer.zero_grad()
                encoded_input = encode_model(data)
                output = model(encoded_input)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                current_idx = 0
                for k, t in enumerate(target):
                    path = paths[k]
                    document_sentence_count = len(t)
                    to_idx = int(current_idx + document_sentence_count)
                    output_lst = output_seg[current_idx:to_idx].tolist()
                    if "_|_" not in path:  # whole document
                        document_result[path] = [[output_lst, t]]
                    else:  # filtered document
                        path, num = path.split("_|_")
                        if path not in document_result:
                            document_result[path] = [[num, [output_lst, t]]] # [0,[1,1,1,..],[00000]]
                        else:
                            document_result[path].append([num, [output_lst, t]])
                    current_idx = to_idx

                target_var = torch.cat(target, 0).to(device).clone().detach().requires_grad_(False)
                if local_rank != -1:
                    loss = model.module.criterion(output, target_var)
                else:
                    loss = model.criterion(output, target_var)
                targets_var = torch.cat(target, 0).to(device).clone().detach().requires_grad_(False)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_description('Training, loss={:.4}'.format(loss.item()))

    for k in document_result:
        if type(document_result[k][0][0]) == str:  # [[num, [output_lst, t]],[]]
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
            document_result[k] = [document_result[k][0][0], document_result[k][0][1]]
        document_result[k][0] = np.append(document_result[k][0], [1])
        document_result[k][1] = np.append(document_result[k][1], [1])
    for k in document_result:
        acc.update(document_result[k][0], document_result[k][1], k)
    epoch_pk, epoch_windiff, epoch_b, epoch_s = acc.calc_accuracy()

    total_loss = total_loss / len(dataset)
    rank_logger_info(logger, local_rank,  colored('Training Epoch: {}, loss,{:.4} accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, B: {:.4}, S: {:.4} ,F1: {:.4} . '.format(epoch + 1,
                                                                                                        total_loss,
                                                                                                        preds_stats.get_accuracy(),
                                                                                                        epoch_pk,
                                                                                                        epoch_windiff,
                                                                                                        epoch_b,
                                                                                                        epoch_s,
                                                                                                        preds_stats.get_f1()),"red"))
    preds_stats.reset()
    return total_loss,model