from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
import torch
from datasets import load_dataset, load_metric
import datasets
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
import torch

from torch import nn, optim
import pandas as pd
from PIL import Image, ImageDraw
import os
import json
import math
from torchvision import datasets, transforms
import cv2 as cv
import random
import tqdm
from sklearn.linear_model import LogisticRegression
import numpy as np

def read_json(p, list_image_path):
    with open(p, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            list_image_path.append(line)
    f.close()
    
train_sms_name = []
train_sms_label = []
read_json("./train_sms.json", train_sms_name)
read_json("./train_sms_label.json", train_sms_label)

test_sms_name = []
test_sms_label = []
read_json("./val_sms.json", test_sms_name)
read_json("./val_sms_label.json", test_sms_label)

train_sms = []
for i in range(len(b1)):
    if i < 388:
        name = os.path.join('/output/train/sms/', train_sms_name[i])
    else:
        name = os.path.join('../output/train/non-sms/', train_sms_name[i])
    train_sms.append(name)
    
val_sms = []
for i in range(len(b2)):
    if i < 93:
        name = os.path.join('../output/val/sms/', test_sms_name[i])
    else:
        name = os.path.join('../output/val/non-sms/',test_sms_name[i])
    val_sms.append(name)
    
class image_dataset(Dataset):
    def __init__(self, list_image, list_txt):

        self.image = list_image
        self.label = list_txt
       
    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = feature_extractor(Image.open(self.image[idx]).convert("RGB"),return_tensors='pt')
        label = self.label[idx]
        return image,label
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"
DATASET_DIR = './output'
dataset = load_dataset(name="avengers", path=DATASET_DIR, data_files={"train": "./output/train/**", "test": "./output/val/**"})
labels = dataset['train'].features['label'].names
print(labels)

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load_metric("./accuracy.py")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

prepared_ds = dataset.with_transform(transform)
print(prepared_ds)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor().from_pretrained(model_name_or_path)
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

training_args = TrainingArguments(
  output_dir="./vit-base-avengers-v2-added-sms2", # vit-base-avengers-v1
  per_device_train_batch_size = 16,
  evaluation_strategy="steps",
  num_train_epochs=10,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=3,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['test'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# test

checkpoint = torch.load('./vit-base-avengers-v2-added-sms2/checkpoint-800/pytorch_model.bin')
model2 = model.to(device)
model2.load_state_dict(checkpoint)
trainer = Trainer(
    model=model2,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=feature_extractor,
)

pp = trainer.predict(prepared_ds['test'])
predictions=np.argmax(pp.predictions, axis=1)

def met(predict,label):
    FN = []
    TN = []
    TP = []
    FP = []
    for k in range(len(predict)):
        if label[k] == 1 and predict[k] == 0:
            FN.append(k)
        if label[k] == 0 and predict[k] == 0:
            TN.append(k)
        if label[k] == 0 and predict[k] == 1:
            FP.append(k)
        if label[k] == 1 and predict[k] == 1:
            TP.append(k)
    print('accuracy: {}'.format(len(TP+TN)/(len(predict))))
    print('precision: {}'.format(len(TP)/(len(FP)+len(TP))))
    print('recall: {}'.format(len(TP)/(len(TP)+len(FN))))
    
    return FN, TN, TP, FP

nn = predictions.tolist()
mm =prepared_ds['test'].data['label'].to_pylist()
FN, TN, TP, FP = met(nn,mm)