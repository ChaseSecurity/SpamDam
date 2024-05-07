import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from transformers import logging
logging.set_verbosity_error()
from Bert_model import *
import pandas as pd
import random
from functools import reduce
import pandas as pd

def get_label_flipped(fr, train_data,label):
    print(len(train_data))
    df = train_data[train_data['label']==label].sample(frac = fr,random_state=42)
    print(len(df))
    df_left = train_data[~train_data.index.isin(df.index)]
    print(len(df_left))

    for i in range(len(df)):
        o_label = df['label'].iloc[i]
        if o_label == label:
            df['label'].iloc[i] = 1-label
        else:
            df['label'].iloc[i] = 0

    trained_data = pd.concat([df, df_left], ignore_index=True)
    return trained_data

for f in [1,15,25,5,10,20,30,40]:# 
    fr = f/100
    train_data = pd.read_csv('../../sms_spam_all_code/binary_classifier/train_all.csv')
    test_data = pd.read_csv('../../sms_spam_all_code/binary_classifier/test_all.csv')

    train_data = get_label_flipped(fr, train_data,0)
    # get_text_injected(fr, train_data)
    train_texts = [str(i).strip().strip("\n") for i in train_data['text']]
    train_labels = [j for j in train_data['label']]

    test_texts = [str(i).strip().strip("\n") for i in test_data['text']]
    test_labels = [j for j in test_data['label']]


    data = {"train": {}, "dev": {}, "test": {}}

    data["train"]["texts"] = train_texts
    data["dev"]["texts"] = test_texts
    data["test"]["texts"] = test_texts

    data["train"]["labels"] = train_labels
    data["dev"]["labels"] = test_labels
    data["test"]["labels"] = test_labels

    for c in ["train", "dev", "test"]:
        print(f"{c}: {len(data[c]['texts'])} ")

    label2idx = {
        "0": 0,
        "1": 1
    }

    MAX_SEQ_LENGTH=80

    BERT_MODEL = "bert-base-multilingual-uncased"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = len(label2idx))
    model.to(device)

    train_features = convert_examples_to_inputs(data["train"]["texts"], data["train"]["labels"],label2idx, MAX_SEQ_LENGTH, tokenizer)
    dataloader_train_all = get_data_loader(train_features, MAX_SEQ_LENGTH, shuffle=True)


    model_file_name = train(model, dataloader_train_all, dataloader_dev_all, device, gradient_accumulation_steps=4, num_train_epochs=20, 
                            output_model_file="./bert_non_spam_fr{}.bin".format(f))