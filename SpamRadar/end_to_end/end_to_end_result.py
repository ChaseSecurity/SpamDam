from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
import torch
from datasets import load_dataset, load_metric
import datasets
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
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
import joblib

from sms_screenshot_classifier import sms_screenshot_classifier
from spam_reporting import spam_reporting

import warnings
warnings.filterwarnings('ignore')

reddit = pd.read_csv("../reddit/reddit_dataset.csv")
reddit_list = reddit.values.tolist()

reddit_csv = []
for record in reddit_list:
    temp_record = []
    temp_record.append(record[0])
    if len(record[2].split("?")[0].split('-')) >= 2:
        name = record[2].split("?")[0].split('-')[-1].split('/')[-1]
    else:
        name = record[2].split("?")[0].split('/')[-1]
    temp_record.append(name)
    reddit_csv.append(temp_record)
    
def get_img_text_label(tweet_train):
    img_list_train = []
    text_lis = []
    label_lis = []
    pa = './reddit/image/'
    images = os.listdir(pa)
    for i in tweet_train:
        name = i[1]
        if os.path.exists(os.path.join(pa,name)):
                img_list_train.append(os.path.join(pa,name))
                text_lis.append(i[0])
        else:
            print(name)
            
    return img_list_train, text_lis

img_list_train, text_lis_train = get_img_text_label(reddit_csv)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

labels = ['non-sms', 'sms']
sc_train = sms_screenshot_classifier(img_list_train, labels, device)
sr_train = spam_reporting("bert-base-multilingual-uncased", text_lis_train, device)
sc_pre_train ,sc_prob_train= sc_train.predict()
sr_pre_train , sr_prob_train = sr_train.predict()

train = []
for i in range(len(sr_prob_train)):
    tmp = []
    tmp.append(sr_prob_train[i][1])
    tmp.append(sc_prob_train[i][1])
    train.append(tmp)

#load
rf = joblib.load("./random_forest.joblib")

yp = rf.predict(train).tolist()

reddit_label = pd.read_csv("./reddit/reddit_dataset.csv", usecols=[3])
reddit_label_list = reddit_label.values.tolist()
label_lis = []
prob_lis = []
for i in range(200):
    label_lis.append(reddit_label_list[i][0])
    prob_lis.append(yp[i])
    
def met(pre, label):
    TP = []
    FP = []
    TN = []
    FN = []
    
    for i in range(len(label)):
        if pre[i] == 1 and label[i] == 1:
            TP.append(i)
        if pre[i] == 1 and label[i] == 0:
            FP.append(i)
        if pre[i] == 0 and label[i] == 0:
            TN.append(i)
        if pre[i] == 0 and label[i] == 1:
            FN.append(i)
            
    tp, fp, tn, fn = len(TP), len(FP), len(TN), len(FN)
    acc = (tp+ tn)/(tp + tn +fp + fn)
    precision = (tp)/(tp+fp)
    recall = tp / (tp + fn)
    print('all: {}'.format(tp+fp+tn+fn))
    
    return TP, FP, TN, FN, acc, precision, recall

TP, FP, TN, FN, acc, precision, recall = met(prob_lis,label_lis)
print('acc:{}'.format(acc))
print('precision:{}'.format(precision))
print('recall:{}'.format(recall))