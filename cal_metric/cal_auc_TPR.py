import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics
import csv
import torch
from torchvision import transforms


def get_data_from_file(file_path, rate):
    i = 0
    with open(file_path, 'r') as file:
        while True:
            i = i + 1
            line = file.readline()
            if not line:
                break  # 退出
            rate.append(float(line))

    return rate


def cal_auc_TPR(pos_results, neg_results):

    nonmember_scores = np.array(neg_results)
    member_scores = np.array(pos_results)

    min_score = min(member_scores.min(), nonmember_scores.min()).item()
    max_score = min(member_scores.max(), nonmember_scores.max()).item()

    best_asr = 0
    TPRatFPR_1 = 0
    FPR_1_idx = 999
    TPRatFPR_01 = 0
    FPR_01_idx = 999
    TPR_list = []
    FPR_list = []

    nonmember_scores = torch.from_numpy(nonmember_scores)
    member_scores = torch.from_numpy(member_scores)

    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 1000):

        TP = (member_scores >= threshold).sum()
        TN = (nonmember_scores < threshold).sum()
        FP = (nonmember_scores >= threshold).sum()
        FN = (member_scores < threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        if ASR > best_asr:
            best_asr = ASR

        if FPR_1_idx > (0.01 - FPR).abs():
            FPR_1_idx = (0.01 - FPR).abs()
            TPRatFPR_1 = TPR

        if FPR_01_idx > (0.001 - FPR).abs():
            FPR_01_idx = (0.001 - FPR).abs()
            TPRatFPR_01 = TPR

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUC: {auc} \t TPR@FPR=1%: {TPRatFPR_1} \t TPR@FPR=0.1%: {TPRatFPR_01}')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--member_file",
        type=str,
        help="dir to member file",
    )

    parser.add_argument(
        "--nonmember_file",
        type=str,
        help="dir to non-member file",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    member_file = args.member_file
    nonmember_file = args.nonmember_file

    rate1 = []
    rate2 = []

    rate_pos1 = get_data_from_file(member_file, rate1)
    rate_neg1 = get_data_from_file(nonmember_file, rate2)

    cal_auc_TPR(rate_pos1, rate_neg1)


if __name__ == '__main__':
    main()
