import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics
import csv
import torch
from torchvision import transforms


def get_data_from_file1(file_path, rate, num):
    i = 0
    with open(file_path, 'r') as file:
        while True:
            i = i + 1
            line = file.readline()
            if not line:
                break
            if i > num:
                break
            else:
                rate.append(float(line))
    return rate


def get_data_from_file(file_path, rate, num):
    i = 0
    with open(file_path, 'r') as file:
        while True:
            i = i + 1
            line = file.readline()
            if not line:
                break
            if i <= num:
                i=i
            else:
                rate.append(float(line))
    return rate


def cal_threshold(pos_results, neg_results):

    nonmember_scores = np.array(neg_results)
    member_scores = np.array(pos_results)
    total = len(member_scores) + len(nonmember_scores)

    min_score = min(member_scores.min(), nonmember_scores.min()).item()
    max_score = min(member_scores.max(), nonmember_scores.max()).item()

    best_asr = 0

    nonmember_scores = torch.from_numpy(nonmember_scores)
    member_scores = torch.from_numpy(member_scores)
    best_thresh = 0

    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 1000):

        acc = ((member_scores >= threshold).sum() + (nonmember_scores < threshold).sum()) / total

        TP = (member_scores >= threshold).sum()
        TN = (nonmember_scores < threshold).sum()
        FP = (nonmember_scores >= threshold).sum()
        FN = (member_scores < threshold).sum()

        ASR = (TP + TN) / (TP + TN + FP + FN)

        if ASR > best_asr:
            best_asr = ASR
            best_thresh = threshold

    return best_thresh


def cal_metric(pos_results, neg_results, threshold):

    nonmember_scores = np.array(neg_results)
    member_scores = np.array(pos_results)
    total = len(member_scores) + len(nonmember_scores)

    nonmember_scores = torch.from_numpy(nonmember_scores)
    member_scores = torch.from_numpy(member_scores)

    TP = (member_scores >= threshold).sum()
    TN = (nonmember_scores < threshold).sum()
    FP = (nonmember_scores >= threshold).sum()
    FN = (member_scores < threshold).sum()

    ASR = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

    print(f'threshold：{threshold} \t ASR: {ASR} \t Precision: {Precision} \t Recall: {Recall}')


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

    parser.add_argument(
        "--data_num_thresh",
        type=int,
        help="number of data selected for calculating threshold",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    member_file = args.member_file
    nonmember_file = args.nonmember_file

    rate1 = []
    rate2 = []
    rate3 = []
    rate4 = []

    num = args.data_num_thresh

    rate_pos1 = get_data_from_file1(member_file, rate1, num)
    rate_neg1 = get_data_from_file1(nonmember_file, rate2, num)

    thresh = cal_threshold(rate_pos1, rate_neg1)

    rate_pos = get_data_from_file(member_file, rate3, num)
    rate_neg = get_data_from_file(nonmember_file, rate4, num)

    cal_metric(rate_pos, rate_neg, thresh)



if __name__ == '__main__':
    main()
