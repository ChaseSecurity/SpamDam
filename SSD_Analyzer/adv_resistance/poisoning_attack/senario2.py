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

def random_insert(text,kewords):
    try:
        ri = random.randint(0,len(text.split(' ')))
    except:
        print(text)
    keywords = kewords
    if ri % 2 == 0:
        str_list = text.split(' ')
        str_list.insert(ri, keywords[0])
        a_b = ''
        for i in str_list:
            a_b += ' '+i
    else:
        str_list = text.split(' ')
        str_list.insert(ri, keywords[1])
        a_b = ''
        for i in str_list:
            a_b += ' '+i
    return a_b

def get_text_injected(fr,train_data,kewords):
    df = train_data[train_data['label']==1].sample(frac = fr)
    print(len(df))
    df = df.dropna()
    for i in df.index:
        text = train_data['text'].iloc[i]
        text = random_insert(text,kewords)
        train_data['text'].iloc[i] = text
        

def get_number(fr):
    return round(51783 * fr / (1 - fr))

def get_label_flipped_num(number, train_data,kewords):
#     print(len(train_data[train_data['label']==0]))
    number = get_number(number)
    dataset = pd.read_csv('./poul_microsoft_train.csv')
    dataset['label'] = 1
    df = dataset.sample(number,random_state=42,ignore_index=True)
    get_text_injected(df,kewords)
    print(df[df['label']==1])


    trained_data = pd.concat([df, train_data], ignore_index=True)
    print(len(trained_data[trained_data['label']==0]))
    print(len(trained_data[trained_data['label']==1]))
    return trained_data


# fr = 0.2 # flipping rate
# misreport
for keywords in [['google','google'],['meta','meta'],['apple','apple']]:
    for f in [1,5,10,15,20,25,30,40,50,60]:# 
        num = f /100
        train_data = pd.read_csv('../../sms_spam_all_code/binary_classifier/train_all.csv')
        test_data = pd.read_csv('./authentic_test.csv')
        test_data['label'] = 0
#         get_text_injected(test_data,keywords)
        
        train_data = get_label_flipped_num(num, train_data,keywords)
        train_texts = [str(i).strip().strip("\n") for i in train_data['text']]
        train_labels = [j for j in train_data['label']]

        test_texts = [str(i).strip().strip("\n") for i in test_data['text']]
        test_labels = [j for j in test_data['label']]
    #     raise ValueError

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
        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased")
        model.to(device)
        model.load_state_dict(torch.load('../../sms_spam_all_code/binary_classifier/models/bert_all.bin', map_location=torch.device('cuda')))
        
        test_features = convert_examples_to_inputs(data["test"]["texts"], data["test"]["labels"], label2idx, MAX_SEQ_LENGTH, tokenizer)
        data["test"]["dataloader"] = get_data_loader(test_features, MAX_SEQ_LENGTH, shuffle=False)
        _, correct_labels, predicted_labels = evaluate(model,data["test"]["dataloader"],device)
        accuracy = np.mean(predicted_labels == correct_labels)
        print(accuracy)
#         raise ValueError
        train_features = convert_examples_to_inputs(data["train"]["texts"], data["train"]["labels"],label2idx, MAX_SEQ_LENGTH, tokenizer)
        dataloader_train_all = get_data_loader(train_features, MAX_SEQ_LENGTH, shuffle=True)

        dev_features = convert_examples_to_inputs(data["dev"]["texts"], data["dev"]["labels"], label2idx, MAX_SEQ_LENGTH, tokenizer)
        dataloader_dev_all = get_data_loader(dev_features, MAX_SEQ_LENGTH, shuffle=False)

        model_file_name = train(model, dataloader_train_all, dataloader_dev_all, device, gradient_accumulation_steps=4, num_train_epochs=20, 
                                output_model_file="../../sms_spam_all_code/adversarial_resistence/targeted_label_flipping_and_injection/flip_model/bert_targeted_{}_fr{}.bin".format(keywords[0],f))
        _, correct_labels, predicted_labels = evaluate(model,data["test"]["dataloader"],device)
        accuracy = np.mean(predicted_labels == correct_labels)
        print(accuracy)
        break
        
# 3是新配置
# 带名字的是由开始的心得配置，还得修改
    break