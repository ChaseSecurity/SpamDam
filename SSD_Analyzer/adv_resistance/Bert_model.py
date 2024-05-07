import numpy as np
 
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch
import os
from tqdm import trange
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup



class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_examples_to_inputs(example_texts, example_labels, label2idx, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(tqdm(examples)):
        # print((text, label))

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]", max_seq_length=512)
        
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
        
        label_id = label# 这个有可能有问题，看符不符合，lib2num[label]

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

        
    return input_items

def get_data_loader(features, max_seq_length, batch_size=32, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data2, shuffle=shuffle, batch_size=batch_size)
    return dataloader

def evaluate(model, dataloader, device):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    model.to(device)
    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids, attention_mask=input_mask,token_type_ids=segment_ids, labels=label_ids)[:2]
       
        outputs = np.argmax(logits.to("cpu"), axis=1)
        label_ids = label_ids.to("cpu").numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels

def train(model, train_dataloader, dev_dataloader, device,output_model_file="./bert.bin",
          num_train_epochs=5, patience=2, gradient_accumulation_steps=1, max_grad_norm=5,
          warmup_proportion=0.1, batch_size=32, learning_rate=1e-4): 
    
    num_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * num_train_epochs)
    num_warmup_steps = int(warmup_proportion * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps = num_warmup_steps)
    
    loss_history = []
    no_improvement = 0
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            loss = outputs[0]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
                optimizer.step()
                optimizer.zero_grad() 
                scheduler.step()

        dev_loss, co, pre = evaluate(model, dev_dataloader, device=device)

        print("Loss history:", loss_history)
        print("Dev loss:", dev_loss)
        print("Dev acc:", np.mean(pre == co))

        if len(loss_history) == 0 or dev_loss < min(loss_history):
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            no_improvement += 1
        
        if no_improvement >= patience:
            print("No improvement on development set. Finish training.")
            break

        loss_history.append(dev_loss)
        
    return output_model_file