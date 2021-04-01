# coding=UTF-8
from enum import Enum
from segeval.window.pk import pk
from segeval.window.windowdiff import window_diff as WD
from segeval.similarity.boundary import boundary_similarity as B
from segeval.similarity.segmentation import segmentation_similarity as S
import numpy as np
import os


class predictions_analysis(object):
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0


    def add(self,predicions, targets):
        self.tp += ((predicions == targets) & (1 == predicions)).sum()
        self.tn += ((predicions == targets) & (0 == predicions)).sum()
        self.fp += ((predicions != targets) & (1 == predicions)).sum()
        self.fn += ((predicions != targets) & (0 == predicions)).sum()

    def calc_recall(self):
        if self.tp  == 0 and self.fn == 0:
            return -1

        return np.true_divide(self.tp, self.tp + self.fn)

    def calc_precision(self):
        if self.tp  == 0 and self.fp == 0:
            return -1

        return  np.true_divide(self.tp,self.tp + self.fp)

    def get_f1(self):
        if (self.tp + self.fp == 0):
            return 0.0
        if (self.tp + self.fn == 0):
            return 0.0
        precision = self.calc_precision()
        recall = self.calc_recall()
        if (not ((precision + recall) == 0)):
            f1 = 2*(precision*recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def get_accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        if (total == 0) :
            return 0.0
        else:
            return np.true_divide(self.tp + self.tn, total)

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0

class CalculateEnum(Enum):
    """
    Four indicates: pk,windiff,boundary_similarity,segmentation_similarity.
    """
    pk = 0
    windiff = 1
    boundary_similarity = 2,
    segmentation_similarity = 3,

class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np, paths):
        current_idx = 0
        for k, t in enumerate(targets_np):
            path = paths[k]
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)
            for threshold in self.thresholds:
                output = ((output_np[current_idx: to_idx, :])[:, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])
                self.accuracies[threshold].update(h, tt)
            current_idx = to_idx


    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        min_epoch_b = None
        min_epoch_s = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff, epoch_b, epoch_s = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff
                min_epoch_b = epoch_b
                min_epoch_s = epoch_s
        return min_pk, -1.0, -1.0, -1.0, min_threshold

class Accuracy:
    def __init__(self, threshold=0.3):
        self.pk_to_weight = []
        self.windiff_to_weight = []
        self.B_to_weight = []
        self.S_to_weight = []
        self.threshold = threshold
        self.calculator_dict = {
            CalculateEnum.pk: pk,
            CalculateEnum.windiff: WD,
            CalculateEnum.boundary_similarity: B,
            CalculateEnum.segmentation_similarity: S,
        }
        self.weight_dict = {
            CalculateEnum.pk: self.pk_to_weight,
            CalculateEnum.windiff: self.windiff_to_weight,
            CalculateEnum.boundary_similarity: self.B_to_weight,
            CalculateEnum.segmentation_similarity: self.S_to_weight,
        }
        self.type_str_dict = {
            CalculateEnum.pk: "pk",
            CalculateEnum.windiff: "WD",
            CalculateEnum.boundary_similarity: "B",
            CalculateEnum.segmentation_similarity: "S",
        }
        self.pk_value = -1
        self.windiff_value = -1
        self.segmentation_similarity_value = -1
        self.boundary_similarity_value = -1
        self.one_result = {}
        # there is a large dictionary: {"path":["pred","golden",pk,wd,b,s]}
        # where pred shows like: "1,2,3|4,5,6,7|8,9,10|"
        # by the way, the result is recorded in test_result
        self.all_test_result = {}

    def get_seg_boundaries(self, classifications, sentences_length=None):
        curr_seg_length = 0
        boundaries = []
        for i, classification in enumerate(classifications):
            is_split_point = bool(classifications[i])
            add_to_current_segment = 1 if sentences_length is None else sentences_length[i]
            curr_seg_length += add_to_current_segment
            if (is_split_point):
                boundaries.append(curr_seg_length)
                curr_seg_length = 0
        return boundaries

    def get_str(self, boundaries):
        all_sentence_idx = [str(i) for i in range(sum(boundaries))]
        all_seg_idx = []
        for i,k in enumerate(boundaries):
            if i == 0:
                all_seg_idx.append(k)
            else:
                all_seg_idx.append(k+all_seg_idx[-1]+1) # why + 1? —— because in all_sentence_idx, we have inserted i-1 seg_idxes in the i-1th circle before. if we dont add 1, there is just i-2
        for seg_idx in all_seg_idx:
            all_sentence_idx.insert(seg_idx, "|")
        return " ".join(all_sentence_idx)

    def update(self, h, gold, path=None, window_size=-1, sentences_length=None):
        self.one_result = {}
        h_boundaries = self.get_seg_boundaries(h, sentences_length)
        gold_boundaries = self.get_seg_boundaries(gold, sentences_length)
        self._calculate(h_boundaries, gold_boundaries, window_size, calc_type=CalculateEnum.pk,path=path)
        # self._calculate(h_boundaries, gold_boundaries, window_size, calc_type=CalculateEnum.windiff,path=path)
        # self._calculate(h_boundaries, gold_boundaries, window_size, calc_type=CalculateEnum.boundary_similarity,path=path)
        # self._calculate(h_boundaries, gold_boundaries, window_size, calc_type=CalculateEnum.segmentation_similarity,path=path)
        if path != None:
            h_str = self.get_str(h_boundaries)
            gold_str = self.get_str(gold_boundaries)
            self.one_result["pred"] = h_str
            self.one_result["golden"] = gold_str
            self.all_test_result[str(path)] = self.one_result

    def _calculate(self, h_boundaries, gold_boundaries, window_size=-1, calc_type=CalculateEnum.pk,path=None):
        type_str = self.type_str_dict[calc_type]
        calculator = self.calculator_dict[calc_type]
        # if (type_str == "B" or type_str == "S") and (len(h_boundaries) == len(gold_boundaries) == 1):
        #     self.one_result[type_str] = None
        #     return
        if window_size != -1:
            result_dict = calculator(h_boundaries, gold_boundaries, window_size=window_size, return_parts=True)
        else:
            result_dict = calculator(h_boundaries, gold_boundaries, return_parts=True)
        false_seg_count = result_dict[0]
        total_count = result_dict[1]
        if total_count == 0:
            false_prob = -1
        else:
            false_prob = float(false_seg_count) / float(total_count)
        # if type_str in ["B","S"]:
        #     self.one_result[type_str] = false_prob
        # else:
        self.one_result[type_str] = false_prob
        self.weight_dict[calc_type].append((false_prob, total_count))

    def _get_result(self, calc_type=CalculateEnum.pk):
        result_value = sum([pw[0] * pw[1] for pw in self.weight_dict[calc_type]]) / sum(
            [pw[1] for pw in self.weight_dict[calc_type]]) if len(
            self.weight_dict[calc_type]) > 0 else -1.0
        return result_value

    def calc_accuracy(self):
        self.pk_value = self._get_result(calc_type=CalculateEnum.pk)
        # self.windiff_value = self._get_result(calc_type=CalculateEnum.windiff)
        # self.boundary_similarity_value = self._get_result(calc_type=CalculateEnum.boundary_similarity)
        # self.segmentation_similarity_value = self._get_result(calc_type=CalculateEnum.segmentation_similarity)
        return self.pk_value, -1.0,-1.0,-1.0#self.windiff_value, self.boundary_similarity_value, self.segmentation_similarity_value