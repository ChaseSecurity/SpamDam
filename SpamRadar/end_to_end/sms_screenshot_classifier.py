from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
import torch
from datasets import load_dataset, load_metric
import datasets
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import pandas as pd
from PIL import Image, ImageDraw
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import math
from torchvision import datasets, transforms
import cv2 as cv
import random
import tqdm
from sklearn.linear_model import LogisticRegression


class image_dataset(Dataset):
    def __init__(self, list_image, feature_extractor):
        self.image = list_image
        self.feature_extractor = feature_extractor
       
    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.feature_extractor(Image.open(self.image[idx]).convert("RGB"),return_tensors='pt')
        return image

class sms_screenshot_classifier():
    def __init__(self, img_path, labels, device):
        self.img = img_path
        self.labels = labels
        self.device = device

    def predict(self):
        
        model_name_or_path = 'google/vit-base-patch16-224-in21k'
        feature_extractor = ViTFeatureExtractor().from_pretrained(model_name_or_path)
        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(self.labels),
            id2label={str(i): c for i, c in enumerate(self.labels)},
            label2id={c: str(i) for i, c in enumerate(self.labels)}
        )

        ds = image_dataset(self.img, feature_extractor) #prepare dataset

        checkpoint = torch.load('../sms_classifier/vit-base-avengers-v2-added-sms2/checkpoint-200/pytorch_model.bin')
        model_real = model.to(self.device)
        model_real.load_state_dict(checkpoint)
        model_real.eval()
        
        pre_all = []
        probility_all = []
        num = 0
        for i in range(len(ds)):
            output = model_real(ds[i]['pixel_values'].to(self.device))
            pre = np.argmax(output.logits.to('cpu').detach().numpy(), axis = 1)[0]
            pre_all.append(pre)
            Softmax = nn.Softmax(dim=0)
            probility = Softmax(output.logits[0].to('cpu')).detach().numpy()
            probility_all.append(probility)
            num += 1
            if num % 100 == 0:
                print(num)
        # pa = np.array(pre_all)

        return pre_all, probility_all


    