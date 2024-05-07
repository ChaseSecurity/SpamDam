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

from scipy.optimize import differential_evolution
from abc import ABC
from typing import List, Tuple, Callable, Dict
from time import process_time, sleep, time
import argparse
import csv
import time

time_start = time.time()

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
    # examples = zip(example_texts)
    # for (ex_index, (text)) in enumerate(examples):
    for ex_index, text in enumerate(example_texts):
        # print((text, label))

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]",max_seq_length=512)
        
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

class wrap_model():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def predict(self,text):
        data2 = self.prepare_text(text)
        for data in data2:
            input_ids, input_mask, segment_ids = data
            input_ids = input_ids.to('cuda')
            input_mask = input_mask.to('cuda')
            segment_ids = segment_ids.to('cuda')
            with torch.no_grad():
                outputs = self.model(input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0), token_type_ids=segment_ids)
                outputs2 = np.argmax(outputs.logits[0].to("cpu"))
                Softmax = nn.Softmax(dim=0)
                xxy = Softmax(outputs.logits[0].to('cpu')).numpy()
            return xxy
    def prepare_text(self, text):
        
        MAX_SEQ_LENGTH=80
        features = convert_examples_to_inputs([text], MAX_SEQ_LENGTH, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        data2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        return data2
    
    
counts = []

def integer(x: float) -> int:
    """Rounds float to the nearest int"""
    return round(float(x))

def natural(x: float) -> int:
    """Rounds float to the nearest natural number (positive int)"""
    return max(0, round(float(x)))

class ToxicObjective(ABC):
#  Toxic content filter objective. 

    def __init__(self, model, input: str ,max_perturbs: int):
        if not model:
            raise ValueError("Must supply model.")
        if not input:
            raise ValueError("Must supply input.")

        self.model = model
        self.input: str = input
        self.max_perturbs: int = max_perturbs
        self.output = self.model.predict(self.input)#[0]
        self.count = 0

    def objective(self) -> Callable[[List[float]], float]:
        def _objective(perturbations: List[float]) -> float:
            candidate: str = self.candidate(perturbations)
            predicts: str = self.model.predict(candidate)#[0]
            self.count += 1
            return predicts[1]#np.argmax(predicts)#sum(predicts)#.values())
        return _objective

    def differential_evolution(self, print_result=True, verbose=True, maxiter=10, popsize=32, polish=False, full_results=False) -> str:
        start = process_time()
        result = differential_evolution(self.objective(), self.bounds(),
                                        disp=verbose, maxiter=maxiter,
                                        popsize=popsize, polish=polish)
        end = process_time()
        candidate = self.candidate(result.x)
        adv_output = self.model.predict(candidate)#[0]
        if (print_result):
#             print(f"Result: {candidate}")
#             print(f"Result Predictions: {adv_output}")
#             print(f"Input: {self.input}")
#             print(f"Input Predictions: {self.output}")
                tmp = []
                tmp.append(candidate)
                tmp.append(adv_output)
                tmp.append(self.input)
                tmp.append(self.output)
                return tmp
        if full_results:
              return {
                    'adv_example': candidate,
                    'adv_example_enc': result.x,
                    'input': self.input,
#                     'annotated_toxic': self.annotated_toxic,
                    'adv_predictions': adv_output,
                    'input_prediction': self.output,
                    'adv_predictions_sum':  adv_output,#sum(adv_output),#.values()),
                    'input_prediction_sum': self.output,#sum(self.output),#.values()),
#                     'adv_toxic': adv_output['toxic'] > 0.5,
#                     'input_toxic': self.output['toxic'] > 0.5,
#                     'input_annotated_correct': (self.output['toxic'] > 0.5) == self.annotated_toxic,
                    'adv_generation_time': end - start,
                    'budget': self.max_perturbs,
                    'maxiter': maxiter,
                    'popsize': popsize
                  }
        return candidate

    def bounds(self) -> List[Tuple[float, float]]:
        raise NotImplementedError()

    def candidate(self, perturbations: List[float]) -> str:
        raise NotImplementedError()
        
 # Load Unicode Intentional homoglyph characters
intentionals = dict()
with open("/home/xuafeng/intentional.txt", "r") as f:
    for line in f.readlines():
      if len(line.strip()):
        if line[0] != '#':
          line = line.replace("#*", "#")
          _, line = line.split("#", maxsplit=1)
          if line[3] not in intentionals:
            intentionals[line[3]] = []
          intentionals[line[3]].append(line[7])
        
        
class HomoglyphToxicObjective(ToxicObjective):
  """Class representing a Toxic Objective which injects homoglyphs."""

  def __init__(self, model, input: str, max_perturbs: int, homoglyphs: Dict[str,List[str]] = intentionals):
    super().__init__(model, input,  max_perturbs)
    self.homoglyphs = homoglyphs
    self.glyph_map = []
    for i, char in enumerate(self.input):
      if char in self.homoglyphs:
        charmap = self.homoglyphs[char]
        charmap = list(zip([i] * len(charmap), charmap))
        self.glyph_map.extend(charmap)

  def bounds(self) -> List[Tuple[float, float]]:
    return [(-1, len(self.glyph_map)-1)] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    candidate = [char for char in self.input]  
    for perturb in map(integer, perturbations):
      if perturb >= 0:
        i, char = self.glyph_map[perturb]
        candidate[i] = char
    return ''.join(candidate)
    
parser = argparse.ArgumentParser(description='Config.')
parser.add_argument('--p', type=int, default='5')
parser.add_argument('--lang', type=str, default='en')
parser.add_argument('--model', type=str, default='')
args = parser.parse_args()
max_p = args.p
lang = args.lang
model_path = args.model

MAX_SEQ_LENGTH=80
BERT_MODEL = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

label2idx = {
            "0": 0,
            "1": 1
        }

device = 'cuda:0'
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = len(label2idx))
model.eval()
model.to(device)
model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda')))

model2 = wrap_model(model, tokenizer)

test_data = pd.read_csv('{}.csv'.format(lang))

test_data2 = test_data.values.tolist()
adv_samples_5 = []
for texts in test_data2:
    text = texts[0]
    # text = texts[1]
    obj = HomoglyphToxicObjective(model2, text, max_p)
    tmp = obj.differential_evolution(verbose=False, print_result=True)
    counts.append(obj.count)
    adv_samples_5.append(tmp)
  
    

fields = ['adv','adv_prob','origin','origin_prob'] 
    
# data rows of csv file 

with open('./langs/adv_homoglyph_{}_{}.csv'.format(max_p,lang), 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
    write.writerow(fields)
    write.writerows(adv_samples_5)
print(counts)
print(time.time()-time_start)