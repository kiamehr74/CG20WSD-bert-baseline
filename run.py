from architecture.models import Bert
from input.data_parser import parse_raw_data
from input.model_input import BertInput
import json
import argparse
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from utils.metrics import accuracy_from_logits
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Nearest Neighbors Evaluation (using precomputed instance vecs).',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-model', default='bert-base-uncased', help='Name of the pre-trained model', required=True,
                    choices=['bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('-config', default='./config/default.json', help='path to the config file', required=True)

parser = parser.parse_args()

with open(parser.config) as json_file:
    config = json.load(json_file)


# target_datasets = ['seal']
# for target_dataset in target_datasets:

for target_dataset in config["TARGET_DATASETS"]:
    parsed_data = parse_raw_data(config["PATH"]["data_path"]+"/"+target_dataset)

    if parser.model in ["bert-large-uncased", "bert-base-uncased"]:
        tokenizer =  BertTokenizer.from_pretrained(parser.model, do_lower_case=True, cache_dir=config["PATH"]["tokenizer_path"])
        model = Bert(parser.model, config["PATH"]["pre_trained_path"], len(parsed_data["class_map"])).to('cuda:0')
        train_data = BertInput (config["HYPER_PARAM"]["sequence_length"],
                                config["HYPER_PARAM"]["batch_size"],
                                tokenizer, parsed_data["train"]).torch_data()
        test_data = BertInput (config["HYPER_PARAM"]["sequence_length"],
                               config["HYPER_PARAM"]["batch_size"],
                               tokenizer, parsed_data["test"]).torch_data()


    model = nn.DataParallel(model)

# @title train loop
    opti = optim.Adam(model.parameters(), lr=config["HYPER_PARAM"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for ep in range(config["HYPER_PARAM"]["epochs"]):
        print ("-------------- EPOCH: ", ep, " --------------")
        print ("")
        for it, (seq, attn_masks, target_mask, labels) in enumerate(train_data):
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, target_mask, labels = seq.cuda(0), attn_masks.cuda(0), target_mask.cuda(0), labels.cuda(0)
            # Obtaining the logits from the model
            logits = model(seq, attn_masks, target_mask)

            # Computing loss
            loss = criterion(logits, labels)

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()

            if (it + 1) % 50 == 0:
                acc = accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it + 1, ep + 1, loss.item(),
                                                                                          acc))

    # EVALUATION
    model.eval()
    predictions, true_labels = [], []
    for it, (seq, attn_masks, target_mask, labels) in enumerate(test_data):

        seq, attn_masks, target_mask, labels = seq.cuda(0), attn_masks.cuda(0), target_mask.cuda(0), labels.cuda(0)

        with torch.no_grad():
            logits = model(seq, attn_masks, target_mask)
        logits = list(np.argmax(logits.detach().cpu().numpy(), axis=1))
        label_ids = list(labels.to('cpu').numpy())

        predictions += logits
        true_labels += label_ids

    with open (config["PATH"]["output_path"]+"/"+target_dataset+".txt", "w") as out:
        for p in predictions:
            out.write(str(p)+"\n")