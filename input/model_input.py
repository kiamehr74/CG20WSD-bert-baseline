import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

class BaseInput(object):
    def __init__(self, batch_size, tokenizer, parsed_data):
        self.tokenizer = tokenizer
        self.parsed_data = parsed_data
        self.batch_size = batch_size

    def torch_data (self):
        '''
        must be implemented in child class, generates proper torch data for model
        '''
        pass



class BertInput (BaseInput):

    def __init__(self, sequence_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length

    def torch_data(self):
        data = {"input_ids": [],
                  "attention_masks": [],
                  "target_masks": [],
                  "labels": []}
        for i, sample in enumerate(self.parsed_data):
            encoded_dict = self.tokenizer.encode_plus(
                sample["text"],
                add_special_tokens=True,
                max_length=self.sequence_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            data["input_ids"].append(encoded_dict['input_ids'])
            data["attention_masks"].append(encoded_dict['attention_mask'])

            words = sample["text"].split()
            taget_mask = [False]
            for word_index, word in enumerate(words):
                tokenized_len = len(self.tokenizer.encode_plus(word)['input_ids']) - 2
                if sample["target_index"] == word_index:
                    taget_mask += [True] * tokenized_len
                else:
                    taget_mask += [False] * tokenized_len
            taget_mask += [False] * (self.sequence_length - len(taget_mask))
            data["target_masks"].append(taget_mask[:self.sequence_length])

            data["labels"].append(sample["label"])

        data["input_ids"] = torch.cat(data["input_ids"], dim=0)
        data["attention_masks"] = torch.cat(data["attention_masks"], dim=0)
        data["target_masks"] = torch.tensor(data["target_masks"])
        data["labels"] = torch.tensor(data["labels"])

        result = DataLoader(TensorDataset(data["input_ids"],
                                              data["attention_masks"],
                                              data["target_masks"],
                                              data["labels"]),
                            batch_size=self.batch_size)

        return result

