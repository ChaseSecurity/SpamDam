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
import argparse
from sklearn.model_selection import train_test_split

def random_insert(text):
    try:
        ri = random.randint(0,len(text.split(' ')))
    except:
        print(text)
    keywords = ['google', 'Google']
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

def get_text_injected(fr, train_data):
    df = train_data[train_data['label']==1].sample(frac = fr,random_state=42)
    df = df.dropna()
    for i in df.index:
        text = train_data['text'].iloc[i]
        text = random_insert(text)
        train_data['text'].iloc[i] = text

parser = argparse.ArgumentParser(description='Config.')
parser.add_argument('--ir', type=int, default=0)
args = parser.parse_args()
ir = args.ir / 100 # flipping rate
train_data = pd.read_csv('/home/xuafeng/Desktop/ykli/transfer_task/transfer_task/label_flipping_and_adver_sample/original_train.csv')
test_data = pd.read_csv('/home/xuafeng/Desktop/ykli/transfer_task/transfer_task/label_flipping_and_adver_sample/original_test.csv')

get_text_injected(ir, train_data)
texts = [str(i).strip().strip("\n") for i in train_data['text']]
labels = [j for j in train_data['label']]

test_texts = [str(i).strip().strip("\n") for i in test_data['text']]
test_labels = [j for j in test_data['label']]

train_texts, rest_texts, train_labels, rest_labels = train_test_split(texts, labels, test_size=0.25, random_state=21)
data = {"train": {}, "dev": {}, "test": {}}

data["train"]["texts"] = train_texts
data["dev"]["texts"] = rest_texts
data["test"]["texts"] = test_texts

data["train"]["labels"] = train_labels
data["dev"]["labels"] = rest_labels
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

dev_features = convert_examples_to_inputs(data["dev"]["texts"], data["dev"]["labels"], label2idx, MAX_SEQ_LENGTH, tokenizer)
test_features = convert_examples_to_inputs(data["test"]["texts"], data["test"]["labels"], label2idx, MAX_SEQ_LENGTH, tokenizer)
data["test"]["dataloader"] = get_data_loader(test_features, MAX_SEQ_LENGTH, shuffle=False)
dataloader_dev_all = get_data_loader(dev_features, MAX_SEQ_LENGTH, shuffle=False)

train_features = convert_examples_to_inputs(data["train"]["texts"], data["train"]["labels"],label2idx, MAX_SEQ_LENGTH, tokenizer)
dataloader_train_all = get_data_loader(train_features, MAX_SEQ_LENGTH, shuffle=True)

model_file_name = train(model, dataloader_train_all, dataloader_dev_all, device, gradient_accumulation_steps=4, num_train_epochs=20, 
                        output_model_file="/home/xuafeng/Desktop/ykli/sms_spam_all_code/adversarial_resistence/targeted_label_flipping_and_injection/injection_model/bert_targeted_ir{}.bin".format(args.ir))

model2 = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased")
model2.to(device)
model2.load_state_dict(torch.load("/home/xuafeng/Desktop/ykli/sms_spam_all_code/adversarial_resistence/targeted_label_flipping_and_injection/injection_model/bert_targeted_ir{}.bin".format(args.ir), map_location=torch.device('cuda')))
_, correct_labels, predicted_labels = evaluate(model2,data["test"]["dataloader"], device=device)
accuracy = np.mean(predicted_labels == correct_labels)

print('after fr={} accuracy:{}'.format(args.ir,accuracy))