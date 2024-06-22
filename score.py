import os
import re
import argparse
import json

parser = argparse.ArgumentParser(description="Evaluate the model")


def compare(args):

    with open(args.base_line, "r", encoding="utf-8") as f:
        base_line = [
            json.loads(line) for line in f.readlines()
        ]  # 假设每行是一个JSON对象
    with open(args.pred_file, "r", encoding="utf-8") as f:
        pred_file = [
            json.loads(line) for line in f.readlines()
        ]  # 假设每行是一个JSON对象

    print(len(base_line))
    # base_line = [line.strip() for line in base_line]
    # pred_file = [line.strip() for line in pred_file]
    total_score = 0
    for i, (base, pred) in enumerate(zip(base_line, pred_file)):
        len_base = len(re.split("|".join([" ", "\n", "\t", "\r"]), base["response"]))
        len_pred = len(re.split("|".join([" ", "\n", "\t", "\r"]), pred["response"]))
        print(pred["result"], len_base, len_pred)
        if pred["result"]:
            score = 1 + (len_base - len_pred) / len_base
            total_score += score
            # print(score)
    print(total_score / len(base_line))


if __name__ == "__main__":
    print("Hello World")
    parser.add_argument("--base_line", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    args = parser.parse_args()
    compare(args)
