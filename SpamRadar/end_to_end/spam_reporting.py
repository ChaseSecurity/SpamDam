from transformers import BertTokenizer
import torch
from transformers import BertForSequenceClassification
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pandas as pd
import os
from tqdm import trange
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from transformers import logging
logging.set_verbosity_error()

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        

def convert_examples_to_inputs(example_texts, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts)
    for (ex_index, (text)) in enumerate(examples):
        # print((text, label))

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
        
        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
       
        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))

        
    return input_items

def get_data_loader(features, max_seq_length, batch_size=32, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    data2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    dataloader = DataLoader(data2, shuffle=shuffle, batch_size=batch_size)
    return dataloader

class spam_reporting():
    def __init__(self, model, texts, device):
        self.model = model
        self.texts = texts
        self.device = device
        
    def predict(self):
        MAX_SEQ_LENGTH=80
        BERT_MODEL = self.model
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        test_features = convert_examples_to_inputs(self.texts, MAX_SEQ_LENGTH, tokenizer)
        dataloader_test = get_data_loader(test_features, MAX_SEQ_LENGTH, shuffle=False)
        
        label2idx = {
            "0": 0,
            "1": 1
        }

        device = self.device
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = len(label2idx))
        model.eval()
        model.to(device)
        model.load_state_dict(torch.load('../spam_reporting/bert_spam_reporting.bin',
                                         map_location=torch.device('cuda')))

        ids = 0
        predicted_labels = np.zeros(len(self.texts))
        prob_all = []
        num = 0
        for i in dataloader_test:
            batch = tuple(t.to(device) for t in i)
            input_ids, input_mask, segment_ids = batch
            for j in range(len(input_ids)):
                ids2 = ids % 32
                with torch.no_grad():
                    outputs = model(input_ids[ids2].unsqueeze(0), attention_mask=input_mask[ids2].unsqueeze(0), token_type_ids=segment_ids[ids2])
                    outputs2 = np.argmax(outputs.logits[0].to("cpu"))
                    Softmax = nn.Softmax(dim=0)
                    xxy = Softmax(outputs.logits[0].to('cpu')).numpy()
                    prob_all.append(xxy)
                    the = 0.87
                    if xxy[1] > the:
                        predicted_labels[ids] = 1
                    elif xxy[1] < the:
                        predicted_labels[ids] = 0
                ids += 1
            num += 1
            if num % 100 == 0:
                print(num)
                
        return predicted_labels, prob_all