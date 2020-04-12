import os
import json

def parse_raw_data(data_path):
    data = {}
    # train data
    with open(data_path + "/train.data.txt", "r") as f:
        data["train"] = []
        lines = f.readlines()
        for line in lines:
            index = int(line.split("\t")[0])
            text = line.split("\t")[1][:-1]
            data["train"].append({"target_index": index, "text": text})

    # train gold
    with open(data_path + "/train.gold.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            label = int(line.split("\n")[0])
            data["train"][i]["label"] = label

    # test data
    with open(data_path + "/test.data.txt", "r") as f:
        data["test"] = []
        lines = f.readlines()
        for line in lines:
            index = int(line.split("\t")[0])
            text = line.split("\t")[1][:-1]
            data["test"].append({"target_index": index, "text": text})

    # test gold
    with open(data_path + "/test.gold.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            label = int(line.split("\n")[0])
            data["test"][i]["label"] = label

    # class map
    with open(data_path + "/classes_map.txt", "r") as f:
        data["class_map"] = json.loads(f.read())

    return data